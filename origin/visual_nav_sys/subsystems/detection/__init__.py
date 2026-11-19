"""Object detection subsystem"""
from .object_tracker import extract_object_info, estimate_guidance_from_objects

__all__ = ['extract_object_info', 'estimate_guidance_from_objects']
