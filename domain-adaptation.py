from google.colab import drive
drive.mount('/content/drive')

class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'source':
            root_dir = '/content/drive/My Drive/UCF6'
            output_dir = '/content/drive/My Drive/source/'
            return root_dir, output_dir

        elif database == 'target':
            root_dir = '/content/drive/My Drive/Dann_dataset_converted'
            output_dir = '/content/drive/My Drive/target/'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return '/content/drive/My Drive/c3d-pretrained.pth'

import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.autograd import Function
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from dataset.data_loader import GetLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms


class VideoDataset(Dataset):
    def __init__(self, dataset='source', split='train', clip_len=16, preprocess=False):
        #print(clip_len)
        #print(split)
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        #print(folder)
        self.clip_len = clip_len
        self.split = split

        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        if not self.check_integrity():
            raise RuntimeError('Dataset not found.')

        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()


        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            print(os.listdir(folder))
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names and indices
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "source":
            if not os.path.exists('/content/drive/My Drive/dataloaders/ucf_labels.txt'):
                with open('/content/drive/My Drive/dataloaders/ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == "target":
            if not os.path.exists('/content/drive/My Drive/dann_labels.txt'):
                with open('/content/drive/My Drive/dann_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        capture.release()

    def randomflip(self, buffer):
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        if (buffer.shape[0]<=16):
            time_index=0
        else:
            time_index = np.random.randint(buffer.shape[0] - clip_len)
        #time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer


class C3D(nn.Module):

    def __init__(self, pretrained=False):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        #self.fc6 = nn.Linear(8192, num_classes)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2048)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        #self.pool6 = nn.MaxPool1d(2, stride=2)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        #x = self.fc7(x)
        x = self.dropout(x)

        x = self.fc8(x)
        # logits = self.fc6(x)
        logits = self.relu(x)
        #logits = self.pool6(logits)

        return logits
        #return x

    def __load_pretrained_weights(self):
        corresp_name = {
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(Path.model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    #print(b) 
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    b = [model.fc8]
    #print(b)
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k




class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(2048, 1024))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(1024))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(1024, 512))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(512))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(512, 6))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(2048, 1024))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(1024))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(1024, 512))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(512))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(512, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        # input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        # feature = self.feature(input_data)
        # feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(input_data, alpha)
        class_output = self.class_classifier(input_data)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


if __name__ == "__main__":
    source_dataset_name = 'source'
    target_dataset_name = 'target'
    source_image_root = os.path.join('..', 'dataset', source_dataset_name)
    target_image_root = os.path.join('..', 'dataset', target_dataset_name)
    model_root = os.path.join('..', 'models')
    cuda = True
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 1
    image_size = 28
    n_epoch = 100

    source_train_data = VideoDataset(dataset='source', split='train', clip_len=16, preprocess=False)
    source_train_loader = DataLoader(source_train_data, batch_size=1, shuffle=False, num_workers=0)

    target_train_data = VideoDataset(dataset='target', split='train', clip_len=16, preprocess=False)
    target_train_loader = DataLoader(target_train_data, batch_size=1, shuffle=False, num_workers=0)

    # load model
    c3d_net = C3D(pretrained=True)

    # print(source_c3d_feature_outputs)
    # print(source_c3d_feature_outputs.size())
    # print(target_c3d_feature_outputs)
    # print(target_c3d_feature_outputs.size())

    my_net = CNNModel()

    # setup optimizer
    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if cuda:
        my_net = my_net.cuda()
        # c3d_net = c3d_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    for u in c3d_net.parameters():
        u.requires_grad = False
    
    tb = SummaryWriter()

    # training
    for epoch in range(n_epoch):
        len_dataloader = min(len(source_train_loader), len(target_train_loader))
        
        data_source_iter = iter(source_train_loader)
        data_target_iter = iter(target_train_loader)

        i = 0
        print("len of dataloader", len_dataloader)
        while i < len_dataloader:

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = data_source_iter.next()
            s_img, s_label = data_source
            # if cuda:
            #     # s_img = s_img.cuda()
            #     s_label = s_label.cuda()

            source_c3d_feature_outputs = c3d_net.forward(s_img)

            my_net.zero_grad()
            batch_size = len(s_label)

            # input_img = torch.FloatTensor(batch_size, 3, 16, 112, 112)
            # class_label = torch.LongTensor(batch_size)
            domain_label = torch.zeros(batch_size)
            domain_label = domain_label.long()

            if cuda:
                # s_img = s_img.cuda()
                # s_label = s_label.cuda()
                # input_img = input_img.cuda()
                source_c3d_feature_outputs = source_c3d_feature_outputs.cuda()
                s_label = s_label.cuda()
                # class_label = class_label.cuda()
                domain_label = domain_label.cuda()

            # input_img.resize_as_(s_img).copy_(s_img)
            # class_label.resize_as_(s_label).copy_(s_label)
            
            class_output, domain_output = my_net(input_data=source_c3d_feature_outputs, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # training model using target data
            data_target = data_target_iter.next()
            t_img, _ = data_target
            # print(i)
            # print(t_img.shape)

            # t_img.size()==(4, 3, 16, 112, 112):
            target_c3d_feature_outputs = c3d_net.forward(t_img)
            
            batch_size = len(t_img)

            # input_img = torch.FloatTensor(batch_size, 3, 16, 112, 112)
            domain_label = torch.ones(batch_size)
            domain_label = domain_label.long()

            if cuda:
                # t_img = t_img.cuda()
                #input_img = input_img.cuda()
                target_c3d_feature_outputs = target_c3d_feature_outputs.cuda()
                domain_label = domain_label.cuda()

            #input_img.resize_as_(t_img).copy_(t_img)

            _, domain_output = my_net(input_data=target_c3d_feature_outputs, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()

            i += 1

            tb.add_scalar('Error - Source label', err_s_label.cpu().data.numpy(), epoch)
            tb.add_scalar('Error - Source domain', err_s_domain.cpu().data.numpy(), epoch)
            tb.add_scalar('Error - Target domain', err_t_domain.cpu().data.numpy())

        
            print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
#                 % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                    err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

        torch.save(my_net, '{0}/mnist_mnistm_model_epoch_{1}.pth'.format(model_root, epoch))
        test(source_dataset_name, epoch)
        test(target_dataset_name, epoch)

    print('done')
