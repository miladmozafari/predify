from predify import predify
from torchvision.models import vgg16
predify(vgg16(), './pvgg16_config.toml')