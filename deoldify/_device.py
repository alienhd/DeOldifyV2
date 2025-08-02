import os
import logging
from enum import Enum
from .device_id import DeviceId

#NOTE:  This must be called first before any torch imports in order to work properly!

logger = logging.getLogger(__name__)

class DeviceException(Exception):
    pass

class _Device:
    def __init__(self):
        self.set(DeviceId.CPU)

    def is_gpu(self):
        ''' Returns `True` if the current device is GPU, `False` otherwise. '''
        return self.current() is not DeviceId.CPU
  
    def current(self):
        return self._current_device

    def set(self, device:DeviceId):     
        if device == DeviceId.CPU:
            os.environ['CUDA_VISIBLE_DEVICES']=''
            logger.info("Device set to CPU")
        else:
            # Check if CUDA is actually available before setting GPU
            try:
                import torch
                if torch.cuda.is_available() and torch.cuda.device_count() > device.value:
                    os.environ['CUDA_VISIBLE_DEVICES']=str(device.value)
                    torch.backends.cudnn.benchmark=False
                    logger.info(f"Device set to GPU {device.value} (CUDA available: {torch.cuda.is_available()})")
                else:
                    logger.warning(f"GPU {device.value} not available. CUDA available: {torch.cuda.is_available() if 'torch' in locals() else 'Unknown'}, device count: {torch.cuda.device_count() if 'torch' in locals() and torch.cuda.is_available() else 0}. Falling back to CPU.")
                    os.environ['CUDA_VISIBLE_DEVICES']=''
                    device = DeviceId.CPU
            except ImportError:
                logger.warning("PyTorch not available, falling back to CPU")
                os.environ['CUDA_VISIBLE_DEVICES']=''
                device = DeviceId.CPU
            except Exception as e:
                logger.warning(f"Error setting GPU device: {e}. Falling back to CPU.")
                os.environ['CUDA_VISIBLE_DEVICES']=''
                device = DeviceId.CPU
        
        self._current_device = device    
        return device

    def get_device_info(self):
        """Get detailed information about the current device setup."""
        info = {
            'current_device': self.current(),
            'is_gpu': self.is_gpu(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        }
        
        try:
            import torch
            info.update({
                'torch_available': True,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'torch_version': torch.__version__
            })
            if torch.cuda.is_available():
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
        except ImportError:
            info.update({
                'torch_available': False,
                'cuda_available': False,
                'cuda_device_count': 0
            })
        except Exception as e:
            info['device_error'] = str(e)
            
        return info