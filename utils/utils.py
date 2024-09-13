import sys, traceback
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d

class Suppressor():

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, exception_type, value, traceback):
        sys.stdout = self.stdout
        if exception_type is not None:
            # Do normal exception handling
            raise Exception(f"Got exception: {exception_type} {value} {traceback}")

    def write(self, x): pass

    def flush(self): pass


def recast_w(znodes,w_array,z):

    wint = interp1d(znodes,w_array,kind='linear',
                    bounds_error=False,fill_value=(w_array[0],w_array[-1]))

    wval = wint(z)


    return wval


def wbinned_setup(info):


    znodes    = info['znodes']
    Nactive   = info['Nactive']
    wsettings = info['wsettings']

    a = np.logspace(-5, 0, 100)

    for ind in range(Nactive):
        info['params']['w_nodes_{}'.format(ind)] = deepcopy(wsettings)
        info['params']['w_nodes_{}'.format(ind)]['latex'] = 'w^N_{}'.format(ind)


    if info['w_to_lambda']:
        info['params'].update({'w_nodes_'+str(ind): -1 for ind in range(Nactive,len(znodes))})
    else:
        info['params'].update({'w_nodes_'+str(ind): 'lambda w_nodes_{}: w_nodes_{}'.format(Nactive-1,Nactive-1) for ind in range(Nactive,len(znodes))})

#    for ind in range(15):
#        info['w_{}'.format(ind)] = {'latex': 'w_{}'.format(ind),
#                                    'value': lambda  w_nodes_0,w_nodes_1,
#                                                     w_nodes_2,w_nodes_3,
#                                                     w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[ind])}


    info['params']['w_0'] = {'latex': 'w\_0',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[0])}

    info['params']['w_1'] = {'latex': 'w\_1',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[1])}
    info['params']['w_2'] = {'latex': 'w\_2',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[2])}
    info['params']['w_3'] = {'latex': 'w\_3',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[3])}
    info['params']['w_4'] = {'latex': 'w\_4',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[4])}
    info['params']['w_5'] = {'latex': 'w\_5',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[5])}
    info['params']['w_6'] = {'latex': 'w\_6',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[6])}
    info['params']['w_7'] = {'latex': 'w\_7',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[7])}
    info['params']['w_8'] = {'latex': 'w\_8',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[8])}

    info['params']['w_9'] = {'latex': 'w\_9',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[9])}
    info['params']['w_10'] = {'latex': 'w\_10',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[10])}
    info['params']['w_11'] = {'latex': 'w\_11',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[11])}
    info['params']['w_12'] = {'latex': 'w\_12',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[12])}
    info['params']['w_13'] = {'latex': 'w\_13',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[13])}
    info['params']['w_14'] = {'latex': 'w\_14',
                                         'value': lambda w_nodes_0,w_nodes_1,
                                                         w_nodes_2,w_nodes_3,
                                                         w_nodes_4,w_nodes_5: recast_w(znodes,[w_nodes_0,w_nodes_1,w_nodes_2,w_nodes_3,w_nodes_4,w_nodes_5],-1+1/np.flip(a)[14])}



    return info
