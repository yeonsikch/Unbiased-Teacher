# 테스트 방법 (--input_path에 test data path만 넣어주시고, 아래 방법대로 평가하시면 됩니다.)
1. `python test.py --img_size 416 --device=0 --batch_size=12 --weight './checkpoint/model_best_eff_b5_a2c_swa.pth' --backbone 'efficientnet-b5' --domain=a2c --input_path {{test data path}}`
2. `python test.py --img_size 416 --device=0 --batch_size=12 --weight './checkpoint/model_best_eff_b5_a4c_swa.pth' --backbone 'efficientnet-b5' --domain=a4c --input_path {{test data path}}`
3. `{{test data path}}` : test를 진행할 이미지 폴더를 입력


(예시)


아래와 같은 구성의 폴더라면 `{{test data path}}` 자리에는 `./echocardiography/test`를 입력한다.
```
--./echocardiography/test

--./echocardiography/test/A2C/

--./echocardiography/test/A4C/
```
4. `./detection-results` 폴더 내에 inference 결과 파일이 생성됨.
