
from operator import add
from functools import reduce
from collections import OrderedDict
import time
import numpy as np
#class to track metadata about the training process
# tracks iteration time, communication time
# initialize the object before the batch loop, add a start and end comm phase around the communication and at a start_iter() call
# in the print stage, finally after the batch loop add a print stats

class stage_meta():
    def __init__(self, name):
        self.name = name
        self.agg_list = [0 for x in range(11)]
        self.time_list = list()
        self.start_time = 0
    
    #collect timings for the phase
    def start_phase(self):
        self.start_time = time.time()
    
    def end_phase(self, iteration):        
        stage_length = time.time() - self.start_time

        self.agg_list[iteration] += stage_length
        self.time_list.append(stage_length)
    
    #calculates the summary stats, the different return signatures is ugly, this should be improved in future work
    def get_summary_stats(self, should_show_avg):
        #if there is no time info, return 0, but the return is different based on if we show averages or not
        if len(self.time_list) == 0:
            if should_show_avg == False:
                return (0, 0, 0)
            else:
                return (0, 0, 0, 0, 0)
            
        total_time = reduce(add, self.time_list)
        max_time = max(self.time_list)
        min_time = min(self.time_list)
        if should_show_avg == False:
            return (total_time, max_time, min_time)
        
        avg_time_for_stage, avg_time_per_phase = self.get_averages()
        return (total_time, max_time, min_time, avg_time_for_stage, avg_time_per_phase)
    
    def get_averages(self):
        if len(self.time_list) == 0:
            return (0, 0)
        avg_time_per_phase = np.mean(np.array(self.time_list))
        
        #we prepopulate the agg_list with 0, so filter out the 0s to not ruin the mean
        non_zero_agg = list(filter(lambda x: x > 0, self.agg_list))
        
        if len(non_zero_agg) == 0:
            return (0, avg_time_per_phase)
        avg_time_for_stage = np.mean(np.array(non_zero_agg))

        return (avg_time_for_stage, avg_time_per_phase)
    
    def print_summary_stats(self, should_show_avg):
        if should_show_avg == False:
            print('{} times total: {} max: {}  min: {} '.format(self.name, *self.get_summary_stats(should_show_avg)))
        else:
            print('{} times total: {} max: {}  min: {} avg per stage time: {} avg round time: {}'.format(self.name, *self.get_summary_stats(should_show_avg)))
    


class itr_meta():
    def __init__(self, skip_first_n_rounds=1):
        self.iter_time_list = list()
        self.skip_first_n_rounds = skip_first_n_rounds
        self.iter_timer = 0

        #the order here determines the order of printing
        self.stage_dict = OrderedDict()
        for stage_name in ('forward', 'backward', 'optimize', 'communication'):
            self.stage_dict[stage_name] = stage_meta(stage_name)
        
        self.start_iter()
    
    #marks the start of a new iteration
    def start_iter(self):
        stopped_timer = self.iter_timer
        self.iter_timer = time.time()
        
        #skip updating the time list if this is the first call to start_itr
        if stopped_timer > 0:
            self.iter_time_list.append(self.iter_timer - stopped_timer)
    

    #mark the start of communication if we are past the rounds we should skip
    def start_phase(self, stage):
        if len(self.iter_time_list) < self.skip_first_n_rounds:
            return
        self.stage_dict[stage].start_phase()
    
    #marks the end of communication if past the rounds we should skip
    def end_phase(self, stage):
        if len(self.iter_time_list) < self.skip_first_n_rounds:
            return

        self.stage_dict[stage].end_phase(len(self.iter_time_list))    
    
    #print out statistics on the iteration and communication time
    def print_time_stats(self, skip_summary=True):
        if len(self.iter_time_list) <= self.skip_first_n_rounds: #make sure we have enough valid info to display
            return

        valid_timings = self.iter_time_list[self.skip_first_n_rounds:]

        if len(valid_timings) > self.skip_first_n_rounds :
            avg_time=np.mean(np.array(valid_timings))
        else:
            avg_time = 0
        
        if skip_summary == False:
            print('iteration times total: {} max: {}  min: {} average:{}'.format(reduce(add,valid_timings), max(valid_timings), min(valid_timings), avg_time))
            for stage in self.stage_dict.values():
                stage.print_summary_stats(len(valid_timings) > self.skip_first_n_rounds)
 
                
        print('iteration average time: {}'.format(avg_time))
        
        #the per stage info is stored in a dictionary keyed on the stage name. The aggregate list contains the per iteration sums of times.
        # To make the timing extensible, we loop through the ordered dictionary to find all of the stages and then zip their aggregate lists together to display
        print('iteration|iteration time|{} time|{} time|{} time|{} time'.format(*[key for key in self.stage_dict.keys()]))
        for idx, per_itr_times in enumerate(zip(self.iter_time_list, *[stage.agg_list for stage in self.stage_dict.values()])):
            
            if idx >= self.skip_first_n_rounds:
                print('{}|{}|{}|{}|{}|{}'.format(idx, *per_itr_times))
