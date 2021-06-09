# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn.functional as F

__all__ = ['CELoss', 'MixCELoss', 'GoogLeNetLoss', 'JSDivLoss', 'MultiLabelLoss']


class Loss(object):
    """
    Loss
    """

    def __init__(self, class_dim=1000, epsilon=None):
        assert class_dim > 1, "class_dim=%d is not larger than 1" % (class_dim)
        self._class_dim = class_dim
        if epsilon is not None and epsilon >= 0.0 and epsilon <= 1.0:
            self._epsilon = epsilon
            self._label_smoothing = True
        else:
            self._epsilon = None
            self._label_smoothing = False

    def _labelsmoothing(self, target):
        if target.shape[-1] != self._class_dim:
            one_hot_target = F.one_hot(target, self._class_dim)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self._epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, self._class_dim])
        return soft_target
    
    def _binary_crossentropy(self, input, target):
        if self._label_smoothing:
            target = self._labelsmoothing(target)
            cost = F.binary_cross_entropy_with_logits(logit=input, label=target)
        else:
            cost = F.binary_cross_entropy_with_logits(logit=input, label=target)
        avg_cost = paddle.mean(cost)
        return avg_cost

    def _crossentropy(self, input, target):
        if self._label_smoothing:
            target = self._labelsmoothing(target)
            input = -F.log_softmax(input, axis=-1)
            cost = paddle.sum(target * input, axis=-1)
        else:
            cost = F.cross_entropy(input=input, label=target) 
        avg_cost = paddle.mean(cost)
        return avg_cost

    def _focal_loss(self, input, target, alpha=0.5, gamma=2.0, epsilon=1e-6):
        # https://blog.csdn.net/weixin_40651515/article/details/105804557
        input = paddle.clip(input, epsilon, 1.0 - epsilon)
        if self._label_smoothing:
            target = self._labelsmoothing(target)
        # Calculate cross entropy
        cross_entropy = -target * paddle.log(input)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * target * paddle.pow((1 - input), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = paddle.sum(loss, axis=1)
        avg_cost = paddle.mean(loss)
        return avg_cost

    def _focal_loss22(self, input, target, alpha=0.25, gamma=2, epsilon=1e-6):
        '''
        https://aistudio.baidu.com/aistudio/projectdetail/340037
            alpha 越大越关注y=1的情况
            gamma 越大越关注不确定的情况
        '''
        if self._label_smoothing:
            target = self._labelsmoothing(target)
        input = paddle.clip(input, epsilon, 1.0 - epsilon)
        # target = paddle.clip(target, epsilon, 1.0 - epsilon)
        loss = -1 * (alpha * paddle.pow((1 - input), gamma) * target * paddle.log(input) +
                     (1 - alpha) * paddle.pow(input, gamma) * (1 - target) * paddle.log(1 - input))
        loss = paddle.sum(loss, axis=1)
        avg_cost = paddle.mean(loss)
        return avg_cost

    def _focal_loss33(self, pred, label, alpha=0.25, gamma=2.0):
        # https://www.pythonf.cn/read/79181
        one_hot = paddle.fluid.layers.one_hot(label, self._class_dim)
        cross_entropy = one_hot * paddle.log(pred)
        cross_entropy = paddle.fluid.layers.reduce_sum(cross_entropy, dim=-1)
        weight = -1.0 * one_hot * paddle.fluid.layers.pow((1.0 - pred), gamma)
        weight = paddle.fluid.layers.reduce_sum(weight, dim=-1)
        ax = alpha * one_hot
        alph = paddle.fluid.layers.reduce_sum(ax, dim=-1)
        cost = alph * weight * cross_entropy
        avg_cost = paddle.mean(cost)
        return avg_cost

    def _kldiv(self, input, target, name=None):
        eps = 1.0e-10
        cost = target * paddle.log(
            (target + eps) / (input + eps)) * self._class_dim
        return cost

    def _jsdiv(self, input, target):
        input = F.softmax(input)
        target = F.softmax(target)
        cost = self._kldiv(input, target) + self._kldiv(target, input)
        cost = cost / 2
        avg_cost = paddle.mean(cost)
        return avg_cost

    def __call__(self, input, target):
        pass
    
    
class MultiLabelLoss(Loss):
    """
    Multilabel loss based binary cross entropy
    """

    def __init__(self, class_dim=1000, epsilon=None):
        super(MultiLabelLoss, self).__init__(class_dim, epsilon)

    def __call__(self, input, target):
        cost = self._binary_crossentropy(input, target)
        return cost


class CELoss(Loss):
    """
    Cross entropy loss
    """

    def __init__(self, class_dim=1000, epsilon=None):
        super(CELoss, self).__init__(class_dim, epsilon)

    def __call__(self, input, target):
        cost = self._crossentropy(input, target)
        # cost = self._focal_loss(input, target)
        # print("helloooooooooooooooooooooooooooooooooooooo")
        # exit()
        return cost


class MixCELoss(Loss):
    """
    Cross entropy loss with mix(mixup, cutmix, fixmix)
    """

    def __init__(self, class_dim=1000, epsilon=None):
        super(MixCELoss, self).__init__(class_dim, epsilon)

    def __call__(self, input, target0, target1, lam):
        cost0 = self._crossentropy(input, target0)
        cost1 = self._crossentropy(input, target1)
        cost = lam * cost0 + (1.0 - lam) * cost1  
        avg_cost = paddle.mean(cost)
        return avg_cost


class GoogLeNetLoss(Loss):
    """
    Cross entropy loss used after googlenet
    """

    def __init__(self, class_dim=1000, epsilon=None):
        super(GoogLeNetLoss, self).__init__(class_dim, epsilon)

    def __call__(self, input0, input1, input2, target):
        cost0 = self._crossentropy(input0, target)
        cost1 = self._crossentropy(input1, target)
        cost2 = self._crossentropy(input2, target)
        cost = cost0 + 0.3 * cost1 + 0.3 * cost2
        avg_cost = paddle.mean(cost)
        return avg_cost


class JSDivLoss(Loss):
    """
    JSDiv loss
    """

    def __init__(self, class_dim=1000, epsilon=None):
        super(JSDivLoss, self).__init__(class_dim, epsilon)

    def __call__(self, input, target):
        cost = self._jsdiv(input, target)
        return cost
