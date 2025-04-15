from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter


class ReLLTrainer(object):
    def __init__(self, encoder, memory=None, memory_source=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.memory_source = memory_source

    def train(self, epoch, data_loader,data_loader_source, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            inputs_source = data_loader_source.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)
            inputs_source, labels_source, indexes_source = self._parse_data(inputs_source)

            # forward
            f_out = self._forward(inputs)
            f_out_source = self._forward(inputs_source)

            loss,outputs = self.memory(f_out, labels)
            loss_source,outputs_source = self.memory_source(f_out_source, labels_source)

            loss += loss_source

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
        
class ReLLIndexTrainer(object):
    def __init__(self, encoder, memory=None, memory_source=None, memory_index=None):
        super(ClusterContrastIndexTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.memory_source = memory_source
        self.memory_index = memory_index

    def train(self, epoch, data_loader,data_loader_source, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            inputs_source = data_loader_source.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)
            inputs_source, labels_source, indexes_source = self._parse_data(inputs_source)

            # forward
            f_out = self._forward(inputs)
            f_out_source = self._forward(inputs_source)

            loss,outputs = self.memory(f_out, labels)
            loss_source,outputs_source = self.memory_source(f_out_source, labels_source)
            loss_index,outputs_index = self.memory_index(f_out, indexes,isIndex=True)
            loss_source_index,outputs_source_index = self.memory_index(f_out_source, indexes_source,epoch,isIndex=True)

            loss += loss_source
            loss += loss_index
            loss += loss_source_index

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
       

class ReLLIndexTrainerWithoutSource(object):
    def __init__(self, encoder, memory=None, memory_source=None, memory_index=None):
        super(ClusterContrastIndexTrainerWithoutSource, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.memory_source = memory_source
        self.memory_index = memory_index

    def train(self, epoch, data_loader, data_loader_source, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            # inputs_source = data_loader_source.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)
            # inputs_source, labels_source, indexes_source = self._parse_data(inputs_source)

            # forward
            f_out = self._forward(inputs)

            loss, outputs = self.memory(f_out, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

class PreTrainer(object):
    def __init__(self, encoder, memory=None):
        super(PreTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)

            loss,out = self.memory(f_out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
