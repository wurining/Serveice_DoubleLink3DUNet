from flask import Flask, send_file, request
from flask_restful import Resource, Api
from evaluate import Evaluate
import numpy as np
import tempfile
import os
import sys

UPLOAD_FOLDER = './tmp'

app = Flask("Service_DoubleLink3DUNet")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app)


class Index(Resource):
    def get(self):
        return {'api_status': 'ok!'}


api.add_resource(Index, '/')

# @app.route('/test', methods = ['POST'])
# def inference_group_a_without_upload():
#     inputfile = request.files['sample']
#     tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], inputfile.filename)
#     inputfile.save(tmp_path)
#     app.logger.debug("upload file...")
#     result_nparray = Evaluate().group_a(tmp_path)
#     app.logger.debug("inference...")
#     with tempfile.NamedTemporaryFile() as outfile:
#         np.save(outfile, result_nparray)
#         return send_file(path_or_file=outfile.name,
#                          as_attachment=True,
#                          download_name="result.npy")


@app.route('/a', methods=['POST'])
def inference_group_a():
    inputfile = request.files['sample']
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], inputfile.filename)
    inputfile.save(tmp_path)
    app.logger.debug("upload file...")
    result_nparray = Evaluate().group_a(tmp_path)
    app.logger.debug("inference...")
    with tempfile.NamedTemporaryFile() as outfile:
        np.save(outfile, result_nparray)
        return send_file(path_or_file=outfile.name,
                         as_attachment=True,
                         download_name="result.npy")


@app.route('/b', methods=['POST'])
def inference_group_b():
    inputfile = request.files['sample']
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], inputfile.filename)
    inputfile.save(tmp_path)
    app.logger.debug("upload file...")
    result_nparray = Evaluate().group_b(tmp_path)
    app.logger.debug("inference...")
    with tempfile.NamedTemporaryFile() as outfile:
        np.save(outfile, result_nparray)
        return send_file(path_or_file=outfile.name,
                         as_attachment=True,
                         download_name="result.npy")


@app.route('/c', methods=['POST'])
def inference_group_c():
    inputfile = request.files['sample']
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], inputfile.filename)
    inputfile.save(tmp_path)
    app.logger.debug("upload file...")
    result_nparray = Evaluate().group_c(tmp_path)
    app.logger.debug("inference...")
    with tempfile.NamedTemporaryFile() as outfile:
        np.save(outfile, result_nparray)
        return send_file(path_or_file=outfile.name,
                         as_attachment=True,
                         download_name="result.npy")


if __name__ == '__main__':
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=5000)
    app.run(debug=True, port=5000, host='0.0.0.0')
    # sys.stdout.flush()
    # print(flush=True)

