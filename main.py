import argparse
import tempfile
import requests
import numpy as np
from evaluate import Evaluate
import json
import os

# 任务状态
BatchTaskStatusCreated = 0
BatchTaskStatusPreparing = 1
BatchTaskStatusRunning = 2
BatchTaskStatusSuccess = 3
BatchTaskStatusCancel = 4
BatchTaskStatusError = 5

# Log msg
Log_no_sample_in_params = 'no sample in params'
Log_no_model_in_params = 'no model in params'
Log_fail_to_download_sample = 'fail to download sample: {url}'
Log_fail_to_inference = 'fail to inference: {model}'
Log_fail_to_upload = 'fail to upload result'
Log_fail_to_update_result = 'fail to update result'
Log_success = 'success'


# 记录日志api
log_api = 'http://{api_host}:{api_port}/api/batchTasks/log'
log_method = 'POST'
log_api = log_api.format(api_host=os.getenv(
    "API_SERVICE_HOST"), api_port=os.getenv("API_SERVICE_PORT"))
# Nginx服务
static_api = 'http://{static_host}:{static_port}/{url}'
static_method = 'GET'
# 更新结果api
update_result_api = 'http://{api_host}:{api_port}/api/batchTasks/{id}/result'
update_result_method = 'PUT'
# 上传文件
upload_file_api = 'http://{api_host}:{api_port}/api/file/upload'
upload_file_method = 'POST'
upload_file_api = upload_file_api.format(api_host=os.getenv(
    "API_SERVICE_HOST"), api_port=os.getenv("API_SERVICE_PORT"))


sess = requests.Session()


def send_log(data):
    print("send_log", req_params, flush=True)
    headers = {
        'Authorization': 'Bearer ' + os.getenv('TOKEN'),
        'Content-Type': 'application/json'
    }
    res = requests.request(log_method, log_api,
                           headers=headers, data=json.dumps(data))
    print(res, flush=True)


def send_log_and_exit(code, data):
    print("send_log_and_exit", code, json.dumps(data), flush=True)
    headers = {
        'Authorization': 'Bearer ' + os.getenv('TOKEN'),
        'Content-Type': 'application/json'
    }
    res = requests.request(log_method, log_api,
                           headers=headers, data=json.dumps(data))
    print(res, flush=True)
    sess.close()
    exit(code)


# log request params
req_params = {
    'task_id': None,
    'params': None,
    'result': None,
    'status': BatchTaskStatusCancel,
    'content': None
}


# 解析命令行参数sample
try:
    args = argparse.ArgumentParser()
    args.add_argument('--id', type=str, help='task id', required=True)
    args.add_argument('--params', type=str, help='params json', required=True)
    print(args.parse_args())
    id = args.parse_args().id
    params = args.parse_args().params
    req_params['task_id'] = id
    req_params['params'] = params

    # prepare api
    update_result_api = update_result_api.format(id=id, api_host=os.getenv(
        "API_SERVICE_HOST"), api_port=os.getenv("API_SERVICE_PORT"))
    req_params['status'] = BatchTaskStatusPreparing
    send_log(req_params)

    # load params
    params_obj = json.loads(params)
    if 'sample' not in params_obj:
        req_params['status'] = BatchTaskStatusError
        req_params['content'] = Log_no_sample_in_params
        send_log_and_exit(1, req_params)
    elif len(params_obj['sample']) == 0:
        req_params['status'] = BatchTaskStatusError
        req_params['content'] = Log_no_sample_in_params
        send_log_and_exit(1, req_params)
    if 'model' not in params_obj:
        req_params['status'] = BatchTaskStatusError
        req_params['content'] = Log_no_model_in_params
        send_log_and_exit(1, req_params)

    samlpe_url = params_obj['sample'][0]
    static_api = static_api.format(url=samlpe_url, static_host=os.getenv(
        "NGINX_SERVICE_HOST"), static_port=os.getenv("NGINX_SERVICE_PORT"))
    print(static_api, flush=True)
    model = params_obj['model']

    # get sample
    samlpe_res = requests.request(static_method, static_api)
    if samlpe_res.status_code != 200:
        req_params['status'] = BatchTaskStatusError
        req_params['content'] = Log_fail_to_download_sample.format(
            url=samlpe_url)
        send_log_and_exit(1)
    with tempfile.NamedTemporaryFile() as sample_file:
        sample_file.write(samlpe_res.content)
        sample_file.flush()
        # get filepath
        sample_file.seek(0)
        sample_file.flush()

        # 模型
        MODELS = {
            'a': Evaluate().group_a,
            'b': Evaluate().group_b,
            'c': Evaluate().group_c,
        }
        # inference
        if model not in MODELS:
            req_params['status'] = BatchTaskStatusError
            req_params['content'] = Log_no_model_in_params.format(model=model)
            send_log_and_exit(1, req_params)

        req_params['status'] = BatchTaskStatusRunning
        send_log(req_params)
        result_nparray: np.ndarray = MODELS[model](sample_file)
        if result_nparray is None:
            req_params['status'] = BatchTaskStatusError
            req_params['content'] = Log_fail_to_inference.format(model=model)
            send_log_and_exit(1, req_params)

    #  upload result
    with tempfile.NamedTemporaryFile() as outfile:
        np.savez_compressed(outfile, arr=result_nparray)
        print("file:", outfile.name, flush=True)
        result_url = requests.request(upload_file_method,
                                      upload_file_api,
                                      headers={
                                          'Authorization': 'Bearer ' + os.getenv('TOKEN'),
                                      },
                                      files={'file': (outfile.name+".npz", outfile, 'application/octet-stream')})
        if result_url.status_code != 200:
            req_params['status'] = BatchTaskStatusError
            req_params['content'] = Log_fail_to_upload
            send_log_and_exit(1, req_params)

        print(result_url.json(), flush=True)
        req_params['result'] = [result_url.json()['data']['rel_path']]
        req_params['status'] = BatchTaskStatusSuccess
        req_params['content'] = Log_success
        # get the url of result
        result_update_url = requests.request(update_result_method, update_result_api,
                                             headers={
                                                 'Content-Type': 'application/json',
                                                 'Authorization': 'Bearer ' + os.getenv('TOKEN'),
                                             },
                                             data=json.dumps({'result': json.dumps(req_params['result'])}))
        if result_update_url.status_code != 200:
            req_params['status'] = BatchTaskStatusError
            req_params['content'] = Log_fail_to_update_result
            send_log_and_exit(1, req_params)
        send_log_and_exit(0, req_params)

except Exception as e:
    import logging
    logging.exception(e)
    req_params['status'] = BatchTaskStatusError
    req_params['content'] = str(e)
    send_log_and_exit(1, req_params)
