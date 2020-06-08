#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import array
import sys
import itertools
import os
import time
import logging
import pandas as pd
import json
import time
import functools
import glob
import numpy as np
import numba
import csv
from struct import unpack
from datetime import timezone
from datetime import datetime
from multiprocessing import Pool
from threading import Lock


# In[2]:


# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

runtime = {}
runtime_lock = Lock()

def join_lists(list_lists):
    return list(itertools.chain.from_iterable(list_lists))


class file_metrics:
    header = {}
    df = pd.DataFrame()

    def __init__(self, file_path):
        self.read_from_file(file_path)

    def read_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            header_length = unpack('I', f.read(4))[0]
            #logger.debug('header_length = %d', header_length)
        
            self.header = json.loads(f.read(header_length))
            self.header['nrays'] = len(self.header['source_file']['modulus']) * 8
            #logging.debug(str(self.header))

            file_length = os.path.getsize(file_path) - header_length - 4
            #logger.debug('File size w/o header is {} B'.format(file_length))

            data = array.array('f')
            data.fromfile(f, file_length // 4)

            npoints = self.header['npoints_zipped']
            nmetrics = len(self.header['metrics'])
            nrays = self.header['nrays']
            nbands = self.header['source_file']['nbands'] + 1


            time_started = time.time()
            struct = {
                'metric_num': [i for i in range(nmetrics)] * (npoints * nrays * nbands),
                'band_num': join_lists([[i] * nmetrics for i in range(nbands)]) * (npoints * nrays),
                'ray_num': join_lists([[i] * nbands * nmetrics for i in range(nrays)]) * npoints,
                'ts': join_lists([[i] * (nmetrics * nrays * nbands) for i in range(npoints)]),

                'value': data
            }

            #for key in struct:
                #logging.debug('%s %d', key, len(struct[key]))

            self.df = pd.DataFrame(struct)
            time_finished = time.time()

            #logger.debug('{0:.2f}s elapsed'.format(time_finished - time_started))


# In[3]:


def update_file(selected_file):
    #logging.debug('selected_file = %s, runtime_file = %s', selected_file, runtime['file'])
    #if int(selected_file) == runtime['file']:
        #print('error')
        #return

    runtime['file'] = int(selected_file)
    #logger.debug('Loading %s', os.path.expanduser(os.path.join(runtime['dir_path'], runtime['files'][runtime['file']])))
    runtime['metrics_obj'] = file_metrics(os.path.expanduser(os.path.join(runtime['dir_path'], runtime['files'][runtime['file']])))


# In[23]:


def get_normal(time, ts):
    return datetime.utcfromtimestamp(get_unix(time, ts)).strftime('%d.%m.%Y %H:%M:%S')

def get_unix(time, ts):
    return datetime.strptime(time, '%d.%m.%Y %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() + ts*10

def get_astro(time, ts):
    return time + ts*10

def save_results(events, runtime):
    date = runtime['metrics_obj'].header['source_file']['datetime']
    star = runtime['metrics_obj'].header['source_file']['star_begin']
    with open('results.csv', 'a') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL)
        for ts in np.sort(list(events.keys())):
            date_star = get_astro(star, ts)
            date_unix = get_unix(date, ts)
            date_normal = get_normal(date, ts)
            events_count = len(events[ts])
            rays = "|".join(str(i) for i in events[ts])
            wr.writerow([date_normal, date_unix, date_star, events_count, rays])


# In[5]:


def chunkify(n, df):
    return np.array_split(df, n)

def reshape_df(df, metrics, rays):
    df = df[df.metric_num.isin(metrics)]
    df = df[df.band_num.isin([0, 1, 2, 3, 4, 5])]
    return df[df.ray_num.isin(rays)].reset_index(drop=True)

@numba.jit
def create_threshold(a, b, n):
    return a + b*n
@numba.jit
def evaluate(a, b):
    return a > b

def find_events_old(data, tup):
    size = int(3)
    n, m = tup
    while not data.empty:
        dfc = data.head(size*7)
        metric_max, metric_med, metric_dev =[[1], [4], [5]] 
        max_val = dfc[np.in1d(dfc['metric_num'], metric_max)]
        
        threshold = create_threshold(dfc[np.in1d(dfc['metric_num'], metric_med)].value.to_numpy(), 
                                     dfc[np.in1d(dfc['metric_num'], metric_dev)].value.to_numpy(), n)
        results = max_val[max_val.value > threshold]
        if (not results.empty) and len(results.index) < m:
            result_statement = 'Found event: ray number {}, timestamp {}.'
            print(result_statement.format(results[:1].ray_num.tolist(), results[:1].ts.tolist()))
        data = data.iloc[size*7:]
    


def find_events(df, parameters):
    found_events = []
    n, m, rays, metrics = parameters
    df_max = df[df.metric_num.isin([1])][['band_num', 'ray_num', 'ts', 'value']].reset_index(drop='true')
    med = df[df.metric_num.isin([4])][['value']].to_numpy()
    dev = df[df.metric_num.isin([5])][['value']].to_numpy()

    threshold = create_threshold(med, dev, n)
    res = evaluate(df_max['value'].to_numpy(), threshold.T).T

    while not res.size == 0:
        if (np.sum(res[:6]) >= m):
            ts = int(df_max[:1].ts)
            ray = int(df_max[:1].ray_num)
            found_events.append([ts, ray])
        df_max = df_max.iloc[6:]
        res = res[6:]
    return found_events

def merge_events(events):
    data = np.array([], int).reshape(0, 2)
    for arr in events:
        if len(arr) != 0:
            data = np.concatenate((data, np.array(arr)))
    return data   

def create_dict(data):
    events = {}
    for event in merge_events(data):
        t, r = event
        if t in events:
            events[t].append(r)
        else:
            events[t] = [r]
    return events


# In[6]:


def load(dir_path):
    runtime['dir_path'] = os.path.expanduser(dir_path)
    if not os.path.exists(runtime['dir_path']):
        logger.error('Directory %s does not exist, exiting', runtime['dir_path'])
        return

    logger.debug(glob.glob(os.path.expanduser(runtime['dir_path'] + "/*.processed")))
    runtime['files'] = list(map(os.path.basename, glob.glob(os.path.expanduser(runtime['dir_path'] + "/*.processed"))))
    if len(runtime['files']) == 0:
        logger.error('List of .processed files is empty, exiting')
        return
    
    logger.debug(runtime['files'])

    runtime['file'] = None
    with runtime_lock:
        update_file(0)


# In[24]:


def run(path, n=5, m=3):
    total_time = time.time()
    load(path)
    rays = [i for i in range(0, 49)]
    metrics = [1, 4, 5]
    parameters = (float(n), float(m), rays, metrics)
    pool = Pool(8)
    with open('results.csv', 'a') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL)
        wr.writerow(['Impulse Events Search. Search parameters: deviance - {}; band quantity - {}'.format(n, m)])
        wr.writerow(['date', 'date_unix', 'date_star', 'ray_count', 'rays'])
        
    
    for file_number in range(0, len(runtime['files'])):
        start_time = time.time()
        with runtime_lock:
            update_file(file_number)
        df = runtime['metrics_obj'].df
        df = reshape_df(df, metrics, rays)
        data_chunks = chunkify(8, df)

        chunked_data = pool.starmap(find_events, zip(data_chunks, itertools.repeat(parameters)))
        events = create_dict(chunked_data)
        save_results(events, runtime)
        print("File ", runtime['files'][file_number], " completed in ", " %s seconds." % (time.time() - start_time))
    
    print("Total time:  %s seconds." % (time.time() - total_time))
    print("Average time: %s seconds." % (time.time() - total_time)/len(runtime['files']))


# In[25]:


dir_path = input("Enter directory path: ")
n = input("Enter deviance: ")
m = input("Enter number of bands: ")
run(dir_path, n, m)


