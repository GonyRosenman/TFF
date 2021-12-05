from torch.optim.lr_scheduler import StepLR

class LrHandler():
    def __init__(self,**kwargs):
        self.final_lr = 1e-5
        self.base_lr = kwargs.get('lr_init')
        self.gamma = kwargs.get('lr_gamma')
        self.step_size = kwargs.get('lr_step')


    def set_lr(self,dict_lr):
        if self.base_lr is None:
            self.base_lr = dict_lr

    def set_schedule(self,optimizer):
        self.schedule = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

    def schedule_check_and_update(self):
        if self.schedule.get_last_lr()[0] > self.final_lr:
            self.schedule.step()
            if (self.schedule._step_count - 1) % self.step_size == 0:
                print('current lr: {}'.format(self.schedule.get_last_lr()[0]))

