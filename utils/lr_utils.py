#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import numpy as np

from chainer.training import extension


class DarknetLRScheduler(extension.Extension):

    """
    1 iteration毎に呼ばれる。
    """

    def __init__(self, attr, rate, init=None, target=None, optimizer=None,
                 step_trigger=None, power=4, burn_in=1000):
        self._attr = attr
        if rate < 0:
            raise ValueError('DarknetLRScheduler does not support negative rate')
        self._rate = rate
        self._init = init
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self.iter = 0
        self._last_value = None
        self.step_trigger = step_trigger
        self.power = power
        self.burn_in = burn_in

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        # ensure that _init is set
        if self._init is None:
            self._init = getattr(optimizer, self._attr)

        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def __call__(self, trainer):
        self.iter += 1
        if self.iter <= self.burn_in:
            optimizer = self._get_optimizer(trainer)
            value = self._init * ((self.iter / self.burn_in) ** self.power)
            self._update_value(optimizer, value)
            return

        if self.iter in self.step_trigger:
            self._t += 1
            optimizer = self._get_optimizer(trainer)
            value = self._init * (self._rate ** self._t)
            if self._target is not None:
                if self._rate > 1:
                    # almost same as value = min(value, self._target), but this
                    # line supports negative values, too
                    if value / self._target > 1:
                        value = self._target
                else:
                    # ditto
                    if value / self._target < 1:
                        value = self._target
            self._update_value(optimizer, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, np.ndarray):
            self._last_value = np.asscalar(self._last_value)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value


class PolynomialShift(extension.Extension):
    """Polynomial Shit """
    def __init__(self, power=0.9, stop_trigger=None, batchsize=4,
                 len_dataset=1, attr='lr'):
        self._attr = attr
        self._power = power
        self._init = None
        self._t = 0
        self._last_value = 0
        if stop_trigger.unit == 'iteration':
            self._maxiter = stop_trigger.period
        elif stop_trigger.unit == 'epoch':
            n_iter_per_epoch = len_dataset / float(batchsize)
            self._maxiter = float(stop_trigger.period * n_iter_per_epoch)

    def initialize(self, trainer):
        optimizer = trainer.updater.get_optimizer('main')
        # ensure that _init is set
        if self._init is None:
            self._init = getattr(optimizer, self._attr)

    def __call__(self, trainer):
        self._t += 1

        optimizer = trainer.updater.get_optimizer('main')
        value = self._init * ((1 - (self._t / self._maxiter)) ** self._power)
        setattr(optimizer, self._attr, value)
        self._last_value = value

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, np.ndarray):
            self._last_value = np.asscalar(self._last_value)
