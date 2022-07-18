# Service_DoubleLink3DUNet

## Model Label
- Double-Link 3D U-Net (Group A)
- Simplify Double-Link 3D U-Net (Group B)
- 3D U-Net (Group C)

## API Endpoints
- `/`
    - ckeck api status
    - GET
    - no param.

- `/a`
    - inference the MRI package file with Group A
    - POST; form-data
    - params: 
        - sample:file
    - curl example: `curl --location --request POST 'http://127.0.0.1:5000/a' --form 'sample=@"./image_BraTS2021_00028.npy"'`

- `/b`
    - inference the MRI package file with Group B
    - POST; form-data
    - params: 
        - sample:file
    - curl example: `curl --location --request POST 'http://127.0.0.1:5000/b' --form 'sample=@"./image_BraTS2021_00028.npy"'`

- `/c`
    - inference the MRI package file with Group C
    - POST; form-data
    - params: 
        - sample:file
    - curl example: `curl --location --request POST 'http://127.0.0.1:5000/c' --form 'sample=@"./image_BraTS2021_00028.npy"'`


## Install Conda Environment

### Export environment.yml
```conda env export > environment.yml```

### Import environment.yml
```conda env create -f environment.yml```

### Starting Flask service
```python app.py```
