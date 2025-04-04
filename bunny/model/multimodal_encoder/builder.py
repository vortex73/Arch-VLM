import os
from .eva_clip.eva_clip_encoder import EvaClipVisionTower
from .siglip.siglip_encoder import SiglipVisionTower, SiglipVisionTowerS2
from .clip.clip_encoder import CLIPVisionTower

# Incase we wanna add future encoders(depends on the scope of the paper)
VISION_ENCODER_REGISTRY = {
    'eva_clip': EvaClipVisionTower,
    'siglip': SiglipVisionTower,
    'siglip_s2': SiglipVisionTowerS2,
    'clip': CLIPVisionTower,
    # Add aimv2 qwen etc
}

def register_vision_encoder(name, encoder_class):
    VISION_ENCODER_REGISTRY[name.lower()] = encoder_class

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    use_s2 = getattr(vision_tower_cfg, 'use_s2', False)
    encoder_type = getattr(vision_tower_cfg, 'encoder_type', None)
    
    if encoder_type and encoder_type.lower() in VISION_ENCODER_REGISTRY:
        encoder_class = VISION_ENCODER_REGISTRY[encoder_type.lower()]
        return encoder_class(vision_tower, args=vision_tower_cfg, **kwargs)
    
    # Old logic as a fail-safe
    if 'sig' in vision_tower.lower():
        if use_s2:
            return SiglipVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'eva' in vision_tower.lower():
        return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'clip' in vision_tower.lower():
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    raise ValueError(f'Unknown vision tower: {vision_tower}. '
                   f'Available encoders: {list(VISION_ENCODER_REGISTRY.keys())}')
