

from roboflow import Roboflow
rf = Roboflow(api_key="RCDp6Nw0sWOrXs7vEl7d")
project = rf.workspace("yaron-lasko-ba2zp").project("calculus-segment")
version = project.version(10)
dataset = version.download("coco-segmentation")
