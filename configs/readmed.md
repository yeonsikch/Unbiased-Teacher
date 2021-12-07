# 테스트 방법
`python test.py --device=0 --batch_size=12 --domain={{all/a2c/a4c}} './checkpoint/model_best_AffineGaussNoise_l.pth' --input_path {{test data path}}`
`{{test_data_path}}` : 예하에 A2C, A4C 폴더를 가진 폴더 경로
`--domain` : a2c data에 대한 성능 측정을 원한다면 a2c, a4c를 원한다면 a4c 입력.

`./detection-results` 폴더 내에 inference 결과 파일이 생성됨.

# 결론 (test data path만 넣어주시고, 아래 방법대로 평가하시면 됩니다.)
1. `python test.py --img_size 416 --device=0 --batch_size=12 --domain=a2c './checkpoint/model_best_eff_b5_a2c.pth' --backbone 'efficientnet-b5' --input_path {{test data path}}`
2. `python test.py --img_size 416 --device=0 --batch_size=12 --domain=a4c './checkpoint/model_best_eff_b5_a4c.pth' --backbone 'efficientnet-b5' --input_path {{test data path}}`