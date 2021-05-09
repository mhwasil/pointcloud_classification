import argparse
from pointcloud_classification import PointCloudClassification

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DGCNN', help='Model (DGCNNC, DGCNN, 3DmFV, SpiderCNN, PointNet, PointNet2)')
parser.add_argument('--train', action='store_true')

config_file = "./config/config.yaml"

if __name__ == '__main__':
    pc_cls = PointCloudClassification(parser.parse_args().model, config_file, train=parser.parse_args().train)
