"""
We modified the code from SSUL

SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""

import math
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import models.model as module_arch
import utils.metric as module_metric
import utils.lr_scheduler as module_lr_scheduler
import data_loader.data_loaders as module_data
import torchvision.transforms as transforms

from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader.task import get_task_labels, get_per_task_classes
from trainer.trainer_voc import Trainer_base, Trainer_incremental
from data_loader import VOC, ADE

from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, DiffusionPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from models.attn_processor import LoRAAttnProcessor, MaskLoRAAttnProcessor, AttnProcessor, MaskAttnProcessor, AttnProcessor2_0, MaskAttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F

def _prepare_device(n_gpu_use, logger):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine,"
                       "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                       "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids
    
def memory_sampling_balanced(config, model, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.to(device)
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.to(device)
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)

    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)
    prev_num_classes = len(old_classes)  # 15
    memory_size = config['data_loader']['args']['memory']['mem_size']
    
    # memory_json = f'./data/{task_dataset}/{task_setting}_{task_name}_memory.json'
    memory_json = config.save_dir.parent / f'step_{task_step}' / f'memory_{memory_size}.json'
    
    if os.path.exists(memory_json):
        return
    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / f'memory_{memory_size}.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]
    else:
        memory_list = {}
        memory_candidates = []

    logger.info("...start memory candidates collection")
    torch.distributed.barrier()
    
    model.eval()
    for batch_idx, data in enumerate(train_loader): # have masked the target
        if task_step > 1:
            with torch.no_grad():
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                
                # WARNING: here, pseudo-labeling seems not useful, as it only used to fruitful the unique labels list
                
                # outputs, _ = model(images, ret_intermediate=False)
                # logit = torch.sigmoid(outputs).detach()
                # pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                # pred_labels += 1
                
                # """ pseudo labeling """
                # targets = torch.where((targets == 0) & (pred_scores >= 0.9), pred_labels.long(), targets.long())
        else:
            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

        for b in range(images.size(0)):
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()
            if 0 in labels:
                labels.remove(0)
            
            memory_candidates.append([img_name, labels]) # mix "inside work dir" with "outside work dir"

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    colorizer = Colorize(21)
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15
    curr_memory_list_new_path = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    np.random.shuffle(sorted_memory_candidates)
    
    random_class_order = old_classes.copy()
    np.random.shuffle(random_class_order)
    num_sampled = 0
    
    while memory_size > num_sampled:
        for cls in random_class_order: # each iter, we find images that contains cls, then go to next cls
            for idx, mem in enumerate(sorted_memory_candidates):
                if len(mem) ==2:
                    img_path, labels = mem
                else:
                    img_path, target_path, labels = mem

                if cls in labels:
                    curr_memory_list[f"class_{cls}"].append(mem)
                    num_sampled += 1
                    del sorted_memory_candidates[idx]
                    break
                    
            if memory_size <= num_sampled:
                break
    
    ##### save the image, target, colored target to the local disk, 
    #### and collect their new path to the 'curr_memory_list_new_path'
    new_img_path = config.save_dir.parent / f'step_{task_step}' / 'collect' / f"{memory_size}"
    new_target_path = config.save_dir.parent / f'step_{task_step}' / 'collect_target' / f"{memory_size}"
    new_target_c_path = config.save_dir.parent / f'step_{task_step}' / 'collect_target_c' / f"{memory_size}"
    for cls in range(1, prev_num_classes + 1):
        cur_list = curr_memory_list[f"class_{cls}"]
        for idx, mem in enumerate(cur_list):
            if len(mem) ==2: # images that are not in current work dir
                img_name, labels = mem
                if task_dataset == 'voc':
                    img_path = train_loader.dataset._image_dir / img_name.split()[0][1:]
                    target_path = train_loader.dataset._cat_dir / img_name.split()[1][1:]
                elif task_dataset == 'ade':
                    img_path = train_loader.dataset._image_dir / f"{img_name}.jpg"#img_name.split()[0][1:]
                    target_path = train_loader.dataset._cat_dir / f"{img_name}.png"#img_name.split()[1][1:]
            else:
                img_path, target_path, labels = mem
            
            if torch.distributed.get_rank() == 0:
                # save image
                img = Image.open(img_path).convert('RGB')
                _img_path = os.path.join(new_img_path, f"{cls}", f"{idx}.jpg" )# e.g., ../memory/voc/step_1/collect/1/0.jpg
                os.makedirs(os.path.dirname(_img_path), exist_ok=True)
                img.save(_img_path)
                # save target, only show the label that is in the prev steps. e.g., 15-1, step 1, we keep the same with the train masking
                target = np.uint8(Image.open(target_path))
                if len(mem) ==2: # target that are not in current work dir, directly open the target which is not masked
                    target = train_loader.dataset.transform_target_masking(torch.from_numpy(target))
                    target = target.numpy() # old train loader, we can safely masking the target and keep the same visiblitiy with the previous step traning
                
                target_c = np.transpose(colorizer(target), (1, 2, 0)).astype(np.uint8)
                
                _target_path = os.path.join(new_target_path, f"{cls}", f"{idx}.png" )
                os.makedirs(os.path.dirname(_target_path), exist_ok=True)
                target = Image.fromarray(target)
                target.save(_target_path)
                # save colored target
                
                target_c = Image.fromarray(target_c)
                _target_c_path = os.path.join(new_target_c_path, f"{cls}", f"{idx}.png" )
                os.makedirs(os.path.dirname(_target_c_path), exist_ok=True)
                target_c.save(_target_c_path)
                
                curr_memory_list_new_path[f"class_{cls}"].append([_img_path, _target_path ,labels])

    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    # sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory
    sampled_memory_list = [mem for mem_cls in curr_memory_list_new_path.values() for mem in mem_cls]  # gather all memory

    # memory_list[f"step_{task_step}"] = {
    #     "memory_candidates": sampled_memory_list,
    #     "memory_list": sorted([mem[0] for mem in sampled_memory_list])
    # }
    memory_list[f"step_{task_step}"] = {
        "cls_wise_memory_candidates": curr_memory_list_new_path,
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list])
    }
    
    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()

def generation_sampling(config, model, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.to(device)
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.to(device)
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)
    
    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)
    prev_num_classes = len(old_classes)  # 15
    memory_size = config['data_loader']['args']['memory']['mem_size']
    if task_dataset == 'voc':
        cls_name_dict = {index: cls_name for index, cls_name in enumerate(VOC)}
    elif task_dataset == 'ade':
        cls_name_dict = {index: cls_name for index, cls_name in enumerate(ADE)}
    # val_composed_transforms = tr.Compose(
    #     [
    #         tr.Resize(size=self.transform_args['crop_size']),
    #         tr.CenterCrop(self.transform_args['crop_size']),
    #         # tr.ToTensor(),
    #         # tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )
    ## Get the memory replay list, will be used for ITI 
    memory_json = config.save_dir.parent / f'step_{task_step}' / f'memory_{memory_size}.json'
    assert os.path.exists(memory_json), f"{memory_json} does not exist"

    with open(memory_json, "r") as json_file:
        memory_replay = json.load(json_file)
    memory_list = memory_replay[f"step_{task_step}"]["memory_list"]
    memory_candidates = memory_replay[f"step_{task_step}"]["memory_candidates"]
    cls_wise_memory_candidates = memory_replay[f"step_{task_step}"]["cls_wise_memory_candidates"]
    if config['replay']['MaskGuide']['use_MaskGuide']:
        if config['replay']['MaskGuide']['Combine']['use_Combine']:
            times = config['replay']['MaskGuide']['Combine']['times']
            MaskGuide_json = config.save_dir.parent / f'step_{task_step}' / f'MaskGuide_{memory_size}_{times-1}X.json'
        else:
            MaskGuide_json = config.save_dir.parent / f'step_{task_step}' / f'MaskGuide_{memory_size}.json'
        if os.path.exists(MaskGuide_json):
            logger.info("***MaskGuide.json already exists, directly use it***")
            return
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            generate_MaskGuide(config, model, train_loader, task_info, logger, gpu, memory_list, cls_wise_memory_candidates, cls_name_dict)
        torch.distributed.barrier()
    # elif config['replay']['MaskMemoryMix']['use_MaskMemoryMix']:
    #     MaskMemoryMix_json = config.save_dir.parent / f'step_{task_step}' / f'MaskMemoryMix_{memory_size}.json'
    #     if os.path.exists(MaskMemoryMix_json):
    #         logger.info("***MaskMemoryMix.json already exists, directly use it***")
    #         return
    #     torch.distributed.barrier()
    #     if torch.distributed.get_rank() == 0:
    #         generate_MaskMemoryMix(config, task_info)
    #     torch.distributed.barrier()
    elif config['replay']['TokenGuide']['use_TokenGuide']:
        TokenGuide_json = config.save_dir.parent / f'step_{task_step}' / f'TokenGuide_{memory_size}.json'
        if os.path.exists(TokenGuide_json):
            logger.info("***TokenGuide.json already exists, directly use it***")
            return
        else:
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                generate_TokenGuide(config, model, train_loader, task_info, logger, gpu, memory_list, cls_wise_memory_candidates, cls_name_dict)
            torch.distributed.barrier()
    # elif config['replay']['TextGuide']['use_TextGuide']:
    #     TokenGuide_json = config.save_dir.parent / f'step_{task_step}' / 'TextGuide.json'
    #     if os.path.exists(TokenGuide_json):
    #         logger.info("***TokenGuide.json already exists, directly use it***")
    #         return
    #     else:
    #         torch.distributed.barrier()
    #         if torch.distributed.get_rank() == 0:
    #             generate_TextGuide(config, model, train_loader, task_info, logger, gpu, memory_list, cls_wise_memory_candidates, cls_name_dict)
    #         torch.distributed.barrier()

def generate_MaskGuide(config, model, train_loader, task_info, logger, device, memory_list, cls_wise_memory_candidates, cls_name_dict):
    MaskGuideArgs = config['replay']['MaskGuide']
    use_replace = config['replay']['MaskGuide']['Replace']['use_Replace']
    use_combine = config['replay']['MaskGuide']['Combine']['use_Combine']
    if use_combine and use_replace:
        raise NotImplementedError
    task_dataset, task_setting, task_name, task_step = task_info
    memory_size = config['data_loader']['args']['memory']['mem_size']

    new_img_path = config.save_dir.parent / f'step_{task_step}' / 'MaskGuide' / f"{memory_size}"
    new_target_path = config.save_dir.parent / f'step_{task_step}' / 'MaskGuide_target' / f"{memory_size}"
    new_target_c_path = config.save_dir.parent / f'step_{task_step}' / 'MaskGuide_target_c' / f"{memory_size}"

    # run inference
    generator = torch.Generator(device=device).manual_seed(task_step)
    
    memory_json = config.save_dir.parent / f'step_{task_step}' / f'memory_{memory_size}.json'
    
    # get the cls_id to step mapping, for extracting right lora and token embedding, which is trained step-wise
    cls_to_mem_step = {}
    for step in range(1, task_step+1):
        with open(memory_json, 'r') as f:
            tmp_replay_dict = json.load(f)
        for cls in tmp_replay_dict[f"step_{step}"]["cls_wise_memory_candidates"].keys():
            class_id = int(cls.split('_')[-1])
            if class_id in cls_to_mem_step:
                continue
            cls_to_mem_step[class_id] = step
    
    with open(memory_json, 'r') as f:
        memory_replay_dict = json.load(f)
    
    if use_combine:
        cls_wise_memory_candidates_X = []
        times = config['replay']['MaskGuide']['Combine']['times']
        assert times > 0
        for time in range(times):
            cls_wise_memory_candidates_X.append({})
    else:
        cls_wise_memory_candidates = {}

    # get the prompts of all seen class from the memory, 
    class_names = []
    for cls, img_list in memory_replay_dict[f"step_{task_step}"]["cls_wise_memory_candidates"].items():
        class_id = int(cls.split('_')[-1])
        class_name = cls_name_dict.get(class_id)
        formatted_name = class_name.replace(" ", "_")
        class_names.append(f"<{formatted_name}>")
    additional_tokens =  {f"{_}": f"<background_{step}_{_}>" for _ in class_names}
    
    # get the train images based on the cls_idx
    cur_step = -1
    for cls, img_list in memory_replay_dict[f"step_{task_step}"]["cls_wise_memory_candidates"].items():
        class_id = int(cls.split('_')[-1])
        
        if use_combine:
            for cls_wise_memory_candidates in cls_wise_memory_candidates_X:
                cls_wise_memory_candidates[f'class_{class_id}'] = []
        else:
            cls_wise_memory_candidates[f'class_{class_id}'] = []
        
        # load ti and lora according to steps
        token_name_list = []
        # if cls_to_mem_step[class_id] != task_step:
        #     continue
        if cls_to_mem_step[class_id] != cur_step:
            lora_step = MaskGuideArgs["lora_step"]
            ti_step = MaskGuideArgs["ti_step"]
            lora_save_dir = str(new_img_path / 'fine-tuned' / f'lora-weights-step-{lora_step}').replace(f'step_{task_step}', f'step_{cls_to_mem_step[class_id]}')
            ti_save_dir = str(new_img_path / 'fine-tuned' / f'learned_embeds-steps-{ti_step}.bin').replace(f'step_{task_step}', f'step_{cls_to_mem_step[class_id]}')
            logger.info(f'load tokens from step {cls_to_mem_step[class_id]}, \n lora_save_dir {lora_save_dir} \n ti_save_dir {ti_save_dir}')
            for k, v in cls_to_mem_step.items():
                if v == cls_to_mem_step[class_id]:
                    logger.info(f'cls {k} from step {v}')
                    token_name_list+= [f"<background_{cls_to_mem_step[class_id]}_<{cls_name_dict.get(k)}>>", f"<{cls_name_dict.get(k)}>"]
            # initialize pipeline step-wise
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                MaskGuideArgs['pretrained_model_path'],
                safety_checker=None,
                revision=None,
                variant=None,
                torch_dtype=torch.float16,
            )
            pipeline = pipeline.to(device)
            
            # load attention processors
            set_mask_attn_processor(pipeline.unet)
            # print(pipeline.unet.attn_processors)
            
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline.set_progress_bar_config(disable=True)
            pipeline.load_lora_weights(str(lora_save_dir))
            # load ti
            tokenizer, text_encoder = load_embeddings(ti_save_dir, token_name_list,  model_path=MaskGuideArgs['pretrained_model_path'], device = device)
            pipeline.tokenizer = tokenizer
            pipeline.text_encoder = text_encoder
            cur_step = cls_to_mem_step[class_id]
        else:
            logger.info(f'has loaded tokens from step {cls_to_mem_step[class_id]}')
        logger.info(f"len of tokenizer: {len(pipeline.tokenizer)}")
        
        logger.info(f'generate images of class id {class_id}')
        for idx, triplet in enumerate(img_list):
            img_path, target_path, label_list = triplet
            ## directly use/save labels
            val_image = Image.open(img_path).convert("RGB")
            val_target = Image.open(target_path)
            sample = {"image": val_image, "label": val_target}
            image_transform, target_transform = train_loader.dataset.transform_mem(sample)
            cls_unique = torch.unique(target_transform.flatten()).cpu().numpy().tolist()
            _target_path = os.path.join(new_target_path, f"{class_id}", f"{idx}.png" )
            os.makedirs(os.path.dirname(_target_path), exist_ok=True)
            target_ = Image.fromarray(target_transform.cpu().numpy().astype(np.uint8))
            target_.save(_target_path)
            colorizer = Colorize(21)
            target_c = np.transpose(colorizer(target_transform.cpu().numpy()), (1, 2, 0)).astype(np.uint8)
            # save colored target
            target_c = Image.fromarray(target_c)
            _target_c_path = os.path.join(new_target_c_path, f"{cls}", f"{idx}.png" )
            os.makedirs(os.path.dirname(_target_c_path), exist_ok=True)
            target_c.save(_target_c_path)
            
            val_prompt, class_ids = prepare_rectify_input(target_transform, task_dataset, class_names, cls_to_mem_step[class_id], cls_name_dict, pipeline.tokenizer)
            
            label = torch.cat([target_transform.unsqueeze(0)] * 2) if MaskGuideArgs['guidance_scale'] > 1 else label.unsqueeze(0)
            class_ids = torch.cat([class_ids.unsqueeze(0)] * 2) if MaskGuideArgs['guidance_scale'] > 1 else class_ids.unsqueeze(0)
            cross_attention_kwargs = {"label": label, "class_ids": class_ids}

            if use_combine:
                with torch.autocast("cuda"):
                    for time in range(times):
                        output = pipeline(val_prompt,
                                    image_transform, 
                                    num_inference_steps=MaskGuideArgs['inference_steps'], 
                                    generator=generator, 
                                    guidance_scale = MaskGuideArgs['guidance_scale'],
                                    strength = MaskGuideArgs['strength'],
                                    cross_attention_kwargs=cross_attention_kwargs).images[0]
                        # Save the image
                        # outputs[-1].save(f"./debug/validation_image_{val_prompt}_diff_{_}.png")
                        # img_transform, _ = train_loader.dataset.transform_mem({"image": outputs[-1], "label": val_target})
                        # img_transform = Image.fromarray(img_transform.cpu().numpy())
                        _img_path = os.path.join(new_img_path, f"{class_id}", f"{idx}_{time}X.png" )
                        os.makedirs(os.path.dirname(_img_path), exist_ok=True)
                        output.save(_img_path)

                        cls_wise_memory_candidates_X[time][f"class_{class_id}"].append([_img_path, _target_path, cls_unique])
            
                for time in range(times):
                    sampled_memory_list = [mem for mem_cls in cls_wise_memory_candidates_X[time].values() for mem in mem_cls]  # gather all memory
                    # save all samples to json
                    MaskGuide_replay_dict = {}
                    MaskGuide_replay_dict[f"step_{task_step}"] = {
                        "cls_wise_memory_candidates": cls_wise_memory_candidates_X[time],
                        "memory_candidates": sampled_memory_list,
                        "memory_list": sorted([mem[0] for mem in sampled_memory_list])
                    }
                    MaskGuide_json = config.save_dir.parent / f'step_{task_step}' / f'MaskGuide_{memory_size}_{time}X.json'
                    with open(MaskGuide_json, "w") as json_file:
                        json.dump(MaskGuide_replay_dict, json_file)
            
            else:
                with torch.autocast("cuda"):
                   
                    output = pipeline(val_prompt,
                                image_transform, 
                                num_inference_steps=MaskGuideArgs['inference_steps'], 
                                generator=generator, 
                                guidance_scale = MaskGuideArgs['guidance_scale'],
                                strength = MaskGuideArgs['strength'],
                                cross_attention_kwargs=cross_attention_kwargs).images[0]
                    # Save the image
                    # outputs[-1].save(f"./debug/validation_image_{val_prompt}_diff_{_}.png")
                    # img_transform, _ = train_loader.dataset.transform_mem({"image": outputs[-1], "label": val_target})
                    # img_transform = Image.fromarray(img_transform.cpu().numpy())
                    _img_path = os.path.join(new_img_path, f"{class_id}", f"{idx}.png" )
                    os.makedirs(os.path.dirname(_img_path), exist_ok=True)
                    output.save(_img_path)

                    cls_wise_memory_candidates[f"class_{class_id}"].append([_img_path, _target_path, cls_unique])
                    sampled_memory_list = [mem for mem_cls in cls_wise_memory_candidates.values() for mem in mem_cls]  # gather all memory
                    # save all samples to json
                    MaskGuide_replay_dict = {}
                    MaskGuide_replay_dict[f"step_{task_step}"] = {
                        "cls_wise_memory_candidates": cls_wise_memory_candidates,
                        "memory_candidates": sampled_memory_list,
                        "memory_list": sorted([mem[0] for mem in sampled_memory_list])
                    }
                    MaskGuide_json = config.save_dir.parent / f'step_{task_step}' / f'MaskGuide_{memory_size}.json'
                    with open(MaskGuide_json, "w") as json_file:
                        json.dump(MaskGuide_replay_dict, json_file)
    
    return
    
def prepare_rectify_input(val_target, task_dataset, class_names, step, cls_name_dict, tokenizer):
    
    unique_cls_id = torch.unique(val_target.flatten())
    if task_dataset == 'voc':
        # add bg tokens and bg prompt
        additional_tokens =  {f"{_}": f"<background_{step}_{_}>" for _ in class_names}
        prompt_names = []
        cls_ids_final = []
    elif task_dataset == 'ade':
        additional_tokens = [f"<background_step{step}_0>"]
        prompt_names = [token for token in additional_tokens]
        cls_ids_final =[0     for   _   in range(len(additional_tokens))]
    else:
        raise NotImplementedError
    

    for cls_id in unique_cls_id:
        if cls_id == 255: 
            continue
        if cls_id == 0: 
            continue
        class_name = cls_name_dict.get(cls_id.item())
        format_name = class_name.replace(" ", "_")
        # each cls token has a corresponding bg token
        if task_dataset == 'voc':
            prompt_names.insert(0, additional_tokens[f"<{format_name}>"])
            cls_ids_final.insert(0, 0)
        prompt_names.append(f"<{format_name}>")
        cls_ids_final.append(cls_id)
    val_prompt = " ".join(prompt_names)
    label = val_target
    class_ids = torch.nn.functional.pad(torch.tensor(cls_ids_final), 
                                        (0, tokenizer.model_max_length - 2 - len(cls_ids_final)), 
                                        value= -1).to(dtype=torch.float32) # tensor of int
    return val_prompt, class_ids

def generate_MaskMemoryMix(config, task_info):
    MaskMemoryMixArgs = config['replay']['MaskMemoryMix']
    task_dataset, task_setting, task_name, task_step = task_info
    memory_size = config['data_loader']['args']['memory']['mem_size']
    memory_json = config.save_dir.parent / f'step_{task_step}' /f'memory_{memory_size}.json'
    MaskGuide_json = config.save_dir.parent / f'step_{task_step}' /f'MaskGuide_{memory_size}_1x.json'
    merge_json = config.save_dir.parent / f'step_{task_step}' /  f'merge_{memory_size}.json'
    with open(memory_json, 'r') as f1, open(MaskGuide_json, 'r') as f2:
        memory_replyay_dict = json.load(f1)
        MaskGuide_replay_dict = json.load(f2)
    merged_memory = {}
    
    merged_memory[f"step_{task_step}"] = {}
    merged_memory[f"step_{task_step}"]['cls_wise_memory_candidates'] = {}
    for class_name in memory_replyay_dict[f"step_{task_step}"]['cls_wise_memory_candidates']:
        merged_memory[f"step_{task_step}"]['cls_wise_memory_candidates'][class_name] = []
        num_records = len(memory_replyay_dict[f"step_{task_step}"]['cls_wise_memory_candidates'][class_name])
        num_records_merged = int(num_records * MaskMemoryMixArgs['merge_ratio'])
        for i in range(num_records_merged):
            img_dir, label_dir, label_list = memory_replyay_dict[f"step_{task_step}"]['cls_wise_memory_candidates'][class_name][i]
            merged_memory[f"step_{task_step}"]['cls_wise_memory_candidates'][class_name].append(
                [img_dir, label_dir, label_list])
        for i in range(num_records - num_records_merged):
            if i >= len(MaskGuide_replay_dict[f"step_{task_step}"]['cls_wise_memory_candidates'][class_name]):
                img_dir, label_dir, label_list = memory_replyay_dict[f"step_{task_step}"]['cls_wise_memory_candidates'][class_name][-i]
            else:
                img_dir, label_dir, label_list = MaskGuide_replay_dict[f"step_{task_step}"]['cls_wise_memory_candidates'][class_name][i]
            merged_memory[f"step_{task_step}"]['cls_wise_memory_candidates'][class_name].append(
                [img_dir, label_dir, label_list])
    sampled_memory_list = [mem for mem_cls in merged_memory[f"step_{task_step}"]['cls_wise_memory_candidates'].values() for mem in mem_cls] 
    merged_memory[f"step_{task_step}"]['memory_candidates'] = sampled_memory_list
    merged_memory[f"step_{task_step}"]['memory_list'] = sorted([mem[0] for mem in sampled_memory_list])
        

    with open(merge_json, "w") as json_file:
        json.dump(merged_memory, json_file)

    return merge_json

def generate_MaskMemoryCombine(config, task_info):
    '''
    MaskMemoryCombineArgs = config['replay']['MaskMemoryCombine']
    task_dataset, task_setting, task_name, task_step = task_info
    memory_size = config['data_loader']['args']['memory']['mem_size']
    
    memory_json = config.save_dir.parent / f'step_{task_step}' /f'memory_{memory_size}.json'
    MaskGuide_json = config.save_dir.parent / f'step_{task_step}' /f'MaskGuide_{memory_size}_{config['times']}x.json'
    combine_json = config.save_dir.parent / f'step_{task_step}' /  f'combine_{memory_size}.json'
    
    with open(memory_json, 'r') as f1, open(MaskGuide_json, 'r') as f2:
        memory_replyay_dict = json.load(f1)
        MaskGuide_replay_dict = json.load(f2)
    combine_memory = {}
    combine_memory[f"step_{task_step}"] = {}
    combine_memory[f"step_{task_step}"]['cls_wise_memory_candidates'] = {}
    
    
    for class_name in memory_replyay_dict[f"step_{task_step}"]['cls_wise_memory_candidates']:
        num_records = len(memory_replyay_dict[f"step_{task_step}"]['cls_wise_memory_candidates'][class_name]) + len(MaskGuide_replay_dict[f"step_{task_step}"]['cls_wise_memory_candidates'][class_name])
        assert num_records == len(memory_replyay_dict[f"step_{task_step}"]['cls_wise_memory_candidates'][class_name]) * (config['times'] + 1)
        combine = []
        for _ in memory_replyay_dict[f"step_{task_step}"]['cls_wise_memory_candidates'][class_name]:
            assert len(_) == 3
            combine.append([_])
        for _ in MaskGuide_replay_dict[f"step_{task_step}"]['cls_wise_memory_candidates'][class_name]:
            assert len(_) == 3
            combine.append([_])
        assert len(combine) == num_records
        combine_memory[f"step_{task_step}"]['cls_wise_memory_candidates'][class_name] = combine
    
    sampled_memory_list = [mem for mem_cls in combine_memory[f"step_{task_step}"]['cls_wise_memory_candidates'].values() for mem in mem_cls] 
    combine_memory[f"step_{task_step}"]['memory_candidates'] = sampled_memory_list
    combine_memory[f"step_{task_step}"]['memory_list'] = sorted([mem[0] for mem in sampled_memory_list])
    
    with open(combine_json, "w") as json_file:
        json.dump(combine_memory, json_file)

    return combine_json
    '''
    pass

def generate_TokenGuide(config, model, train_loader, task_info, logger, device, memory_list, cls_wise_memory_candidates, cls_name_dict):
    # The TokenGuide relies on:
    #  1. images from memory and,
    #  2. finetuned textual token embeddings based on 
    #       (1) a few cls-wise samples from memory replay 
    #       (2) Textual Inversion technique
    TokenGuideArgs = config['replay']['TokenGuide']
    task_dataset, task_setting, task_name, task_step = task_info
    memory_size = config['data_loader']['args']['memory']['mem_size']

    new_img_path = config.save_dir.parent / f'step_{task_step}' / 'TokenGuide' / f"{memory_size}"
    new_target_path = config.save_dir.parent / f'step_{task_step}' / 'TokenGuide_target' / f"{memory_size}"
    new_target_c_path = config.save_dir.parent / f'step_{task_step}' / 'TokenGuide_target_c' / f"{memory_size}"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        TokenGuideArgs['pretrained_model_path'],
        safety_checker=None,
        revision="fp16", 
        torch_dtype=torch.float16
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(task_step)

    memory_json = config.save_dir.parent / f'step_{task_step}' / f'memory_{memory_size}.json'
    with open(memory_json, 'r') as f:
        memory_replay_dict = json.load(f)

    token_emb_dir = TokenGuideArgs['finetuned_token_emb_dir']
    
    cls_wise_memory_candidates = {}
    
    # images from memory
    for cls, img_list in memory_replay_dict[f"step_{task_step}"]["cls_wise_memory_candidates"].items():
        class_id = int(cls.split('_')[-1])
        cls_wise_memory_candidates[f'class_{class_id}'] = []
        
        # get cls name based on cls id
        cls_name = cls_name_dict.get(class_id)
        
        # set textual toekn emb
        embed_path = os.path.join(token_emb_dir ,cls_name ,'learned_embeds.bin')
        tokenizer, text_encoder = load_embeddings(
                embed_path, [cls_name], model_path=TokenGuideArgs['pretrained_model_path'])
        pipe.tokenizer = tokenizer
        pipe.text_encoder = text_encoder
        pipe.set_progress_bar_config(disable=True)
        
        logger.info(f"generate TokenGuide augmented images for class {cls_name} {len(img_list)} memory images")
        for idx, triplet in enumerate(img_list):
            img_path, target_path, label_list = triplet
            ## directly use/save labels
            val_image = Image.open(img_path).convert("RGB")
            val_target = Image.open(target_path)
            # target_transform = np.array(val_target).astype(np.uint8)
            sample = {"image": val_image, "label": val_target}
            image_transform, target_transform = train_loader.dataset.transform_mem(sample)
            cls_unique = torch.unique(target_transform.flatten()).cpu().numpy().tolist()
            _target_path = os.path.join(new_target_path, f"{class_id}", f"{idx}.png" )
            os.makedirs(os.path.dirname(_target_path), exist_ok=True)
            target_ = Image.fromarray(target_transform.cpu().numpy().astype(np.uint8))
            target_.save(_target_path)
            colorizer = Colorize(21)
            target_c = np.transpose(colorizer(target_transform.cpu().numpy()), (1, 2, 0)).astype(np.uint8)
            # save colored target
            target_c = Image.fromarray(target_c)
            _target_c_path = os.path.join(new_target_c_path, f"{cls}", f"{idx}.png" )
            os.makedirs(os.path.dirname(_target_c_path), exist_ok=True)
            target_c.save(_target_c_path)

            # save TokenGuide images
            prompt = f"a photo of a <{cls_name}>" # best quality, extremely detailed, realistic color, natural, daily life
            with torch.autocast("cuda"):
                # default output 512x512
                output = pipe(
                    prompt,
                    image_transform,
                    generator=generator,
                    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                    guidance_scale=TokenGuideArgs['guidance_scale'],
                    strength = TokenGuideArgs['strength'],
                    num_inference_steps=TokenGuideArgs['inference_steps'],
                ).images[0]
                _img_path = os.path.join(new_img_path, f"{class_id}", f"{idx}.png" )
                os.makedirs(os.path.dirname(_img_path), exist_ok=True)
                output.save(_img_path)
            
            # img_transform = val_transform(output)
            # img_transform = Image.fromarray(img_transform)
            # _img_path = new_img_path / f"{cls}" / f"{idx}.jpg"
            # os.makedirs(os.path.dirname(_img_path), exist_ok=True)
            # img_transform.save(_img_path)
        
            ## generate&save pseudo labels
            # transform (e.g., resize, crop) the image before feeding to the model
            # img_transform = img_transform.unsqueeze(0).to(device)
            # target = torch.zeros_like(img_transform).to(device)
            # with torch.no_grad():
            #     outputs, _ = model(img_transform.unsqueeze(0).to(device), ret_intermediate=False)
            #     logit = torch.sigmoid(outputs).detach()
            #     pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
            #     pred_labels += 1
                
            #     """ pseudo labeling """
            #     target = torch.where((target == 0) & (pred_scores >= TokenGuideArgs['pseudo_threshold']), pred_labels.long(), target.long())
            # cls_unique = torch.unique(target.flatten()).detach().cpu().numpy().tolist()
            
            # _target_path = new_target_path / f"{cls}" / f"{idx}.png"
            # os.makedirs(os.path.dirname(_target_path), exist_ok=True)
            # target = Image.fromarray(target.cpu().numpy().squeeze(0).astype(np.uint8))
            # target.save(_target_path)

            cls_wise_memory_candidates[f"class_{class_id}"].append([_img_path, _target_path, cls_unique])

    sampled_memory_list = [mem for mem_cls in cls_wise_memory_candidates.values() for mem in mem_cls]  # gather all memory
    
    # save all samples to json
    TokenGuide_replay_dict = {}
    TokenGuide_replay_dict[f"step_{task_step}"] = {
        "class-wise_memory_candiates": cls_wise_memory_candidates,
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list])
    }
    
    TokenGuide_json = config.save_dir.parent / f'step_{task_step}' / f'TokenGuide_{memory_size}.json'
    with open(TokenGuide_json, "w") as json_file:
        json.dump(TokenGuide_replay_dict, json_file)
    
    return TokenGuide_json


ERROR_MESSAGE = "Tokenizer already contains the token {token}. \
Please pass a different `token` that is not already in the tokenizer."
def load_embeddings(embed_path,
                    token_name_list,
                    model_path = "CompVis/stable-diffusion-v1-4",
                    device = "cuda"):
    
    tokenizer = CLIPTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(
        model_path, 
        subfolder="text_encoder")
    
    # for embed_path in embed_paths:
    for token, token_embedding in torch.load(
            embed_path, map_location="cpu").items():
        print(f"add token {token} to tokenizer")
        # add the token in tokenizer
        num_added_tokens = tokenizer.add_tokens(token)
        assert num_added_tokens > 0, ERROR_MESSAGE.format(token=token)
    
        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))
        added_token_id = tokenizer.convert_tokens_to_ids(token)

        # get the old word embeddings
        embeddings = text_encoder.get_input_embeddings()

        # get the id for the token and assign new embeds
        embeddings.weight.data[added_token_id] = \
            token_embedding.to(embeddings.weight.dtype)

    return tokenizer, text_encoder.to(device)

def set_mask_attn_processor(unet):
    lora_attn_procs = {}
    # unet_lora_parameters = []
    check_keys =list(unet.attn_processors.keys())
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            # if args.train_lora:
            #     lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size)
            # else:
            lora_attn_procs[name] = AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnProcessor()
        else:
            # if args.train_lora:
            #     lora_attn_procs[name] = MaskLoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            # else:
            lora_attn_procs[name] = MaskAttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else MaskAttnProcessor()
        # unet_lora_parameters.extend(lora_attn_procs[name].parameters())
    unet.set_attn_processor(lora_attn_procs)

###############################################colorize segmentation map, code from 
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N, normalized = False):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    if N == 21: # voc
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])
        cmap = cmap/255 if normalized else cmap
        return cmap
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] =  r
            cmap[i, 1] =  g
            cmap[i, 2] =  b
     
    return cmap


class Colorize(object):
    def __init__(self, n=182):
        self.cmap = labelcolormap(n)

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1])) 
     
        for label in range(0, len(self.cmap)):
            mask = (label == gray_image ) 
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image



def generate_TextGuide(config, model, train_loader, task_info, logger, gpu, memory_list, cls_wise_memory_candidates):
    TextGuideArgs = config['replay']['TextGuide']
    
    pipe = StableDiffusionPipeline.from_pretrained(
        TextGuideArgs['pretrained_model_path'],
        revision="fp16", 
        torch_dtype=torch.float16
    ).to(device)
    
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'collect' / 'memory.json'
    with open(memory_json, 'r') as f:
        memory_replay_dict = json.load(f)

    cls_wise_memory_candidates = {}
    for index, cls_id in enumerate(old_classes):
        if cls_id == 0:
            # ignore bg
            continue
        cls_name = cls_name_dict[cls_id]
        cls_prompt = f"a photo of a <{cls_name}>"
        
        # save setting
        new_img_path = config.save_dir.parent / f'step_{task_step}' / 'TokenGuide' / f"{memory_size}"
        new_target_path = config.save_dir.parent / f'step_{task_step}' / 'TokenGuide_target' / f"{memory_size}"
        
        pipe.set_progress_bar_config(disable=True)
        pipe.safety_checker = None
        
        # the number of generated samples of each cls is equal to the memory replay of each cls
        num_generate = len(memory_replay_dict[f"step_{task_step}"]["cls_wise_memory_candidates"][f"class_{cls_id}"])
        for idx in range(num_generate):
            # generate&save TextGuide images
            with autocast(device):
                # default output 512x512
                output = pipe(
                    cls_prompt, 
                    guidance_scale=TextGuideArgs['guidance_scale'],
                    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                    num_inference_steps=50,
                ).images[0]
            img_transform = val_transform(output)
            img_transform = Image.fromarray(img_transform)
            _img_path = new_img_path / f"{cls}" / f"{idx}.jpg"
            os.makedirs(os.path.dirname(_img_path), exist_ok=True)
            img_transform.save(_img_path)

            ## generate&save pseudo labels
            # transform (e.g., resize, crop) the image before feeding to the model
            img_transform = img_transform.unsqueeze(0).to(device)
            target = torch.zeros_like(img_transform).to(device)
            with torch.no_grad():
                outputs, _ = model(img_transform.unsqueeze(0).to(device), ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1
                
                """ pseudo labeling """
                target = torch.where((target == 0) & (pred_scores >= TextGuideArgs['pseudo_threshold']), pred_labels.long(), target.long())
            cls_unique = torch.unique(target.flatten()).detach().cpu().numpy().tolist()
            
            _target_path = new_target_path / f"{cls}" / f"{idx}.png"
            os.makedirs(os.path.dirname(_target_path), exist_ok=True)
            target = Image.fromarray(target.cpu().numpy().squeeze(0).astype(np.uint8))
            target.save(_target_path)

            
            
            cls_wise_memory_candidates[f"class_{cls_id}"].append([_img_path, _target_path, cls_unique])
    
    sampled_memory_list = [mem for mem_cls in cls_wise_memory_candidates.values() for mem in mem_cls]  # gather all memory
    # save all samples to json
    TextGuide_replay_dict = {}
    TextGuide_replay_dict[f"step_{cfg.STEP}"] = {
        "class-wise_memory_candiates": cls_wise_memory_candidates,
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list])
    }
    TextGuide_json = config.save_dir.parent / f'step_{task_step}' / 'TextGuide' /'TextGuide.json'
    with open(TextGuide_json, "w") as json_file:
        json.dump(TextGuide_replay_dict, json_file)
    
    return TextGuide_json

def generate_CannyGuide(config, model, train_loader, task_info, logger, gpu, memory_list, cls_wise_memory_candidates):
    CannyGuideArgs = config['replay']['CannyGuide']
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16) # lllyasviel/control_v11p_sd15_canny, lllyasviel/sd-controlnet-canny
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    pipe.safety_checker = lambda images, clip_input: (images, False)

    memory_json = config.save_dir.parent / f'step_{task_step}' / 'collect' / 'memory.json'
    with open(memory_json, 'r') as f:
        memory_replay_dict = json.load(f)

    new_img_path = config.save_dir.parent / f'step_{task_step}' / 'CannyGuide' / f"{memory_size}"
    new_target_path = config.save_dir.parent / f'step_{task_step}' / 'CannyGuide_target' / f"{memory_size}"
    cls_wise_memory_candidates = {}
    
    # images from MemoryGuide
    for cls, img_list in memory_replay_dict[f"step_{task_step}"]["cls_wise_memory_candidates"].items():
        class_id = int(cls.split('_')[-1])
        cls_wise_memory_candidates[f'class_{cls}'] = []
        
        logger.info(f"generate CannyGuide images for class {cls_name} from {len(img_list)} memory replay images")
        for idx, triplet in enumerate(img_list):
            img_path, target_path, label_list = triplet
            
            # convert memory image to canny
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            edges = cv2.Canny(img, 100, 200)
            edges = np.concatenate([edges, edges, edges], axis=2)
            canny_img = Image.fromarray(edges)
            canny_img_transform = val_transform(canny_img)

            # generate&save CannyGuide images
            for cls_id in label_list:
                cls_name = cls_name_dict[cls_id]
                prompt = prompt + " ".join(f"<{cls_name}>")
            logger.info(f"prompt: {prompt}")
            
            with autocast('cuda'):
                # default output 512x512
                output = pipe(
                    prompt,
                    canny_img_transform,
                    guidance_scale = CannyGuideArgs['guidance_scale'],
                    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                    num_inference_steps=50,
                ).images[0]
            img_transform = val_transform(output) # crop again
            img_transform = Image.fromarray(img_transform.cpu().numpy())
            _img_path = new_img_path / f"{cls}" / f"{idx}.jpg"
            os.makedirs(os.path.dirname(_img_path), exist_ok=True)
            img_transform.save(_img_path)

            ## directly use labels
            target = Image.open(target_path)
            target_transform = val_transform(target) # crop size to 512x512
            cls_unique = torch.unique(target_transform.flatten()).cpu().numpy().tolist()

            _target_path = os.path.join(new_target_path, f"{cls}", f"{idx}.png" )
            os.makedirs(os.path.dirname(_target_path), exist_ok=True)
            target_transform = Image.fromarray(target_transform.cpu().numpy().astype(np.uint8))
            target_transform.save(_target_path)
            
            cls_wise_memory_candidates[f"class_{class_id}"].append([_img_path, _target_path, cls_unique])
    
    sampled_memory_list = [mem for mem_cls in cls_wise_memory_candidates.values() for mem in mem_cls]  # gather all memory
    # save all samples to json
    CannyGuide_replay_dict = {}
    CannyGuide_replay_dict[f"step_{cfg.STEP}"] = {
        "class-wise_memory_candiates": cls_wise_memory_candidates,
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list])
    }
    CannyGuide_json = config.save_dir.parent / f'step_{task_step}' / 'CannyGuide' /'CannyGuide.json'
    with open(CannyGuide_json, "w") as json_file:
        json.dump(CannyGuide_replay_dict, json_file)
    return CannyGuide_json
    
    




    
