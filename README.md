# 레이싱 게임(카트라이더) 자율주행 인공지능 개발
부산대학교 정보컴퓨터공학과 2021년 전기 정보통신대대 팀 졸업과제로 수행한 지도학습을 통한 카트라이더 자율주행 인공지능 개발 프로젝트 저장소입니다.
<br/><br/>

## 구성
* data : 학습 데이터 및 전처리 코드
* model : 모델 구성 파일과 모델 학습 코드
* driver : 모델을 게임에 실행시킬 테스트 코드
* demo : 발표에 사용한 테스트 코드와 모델 가중치
<br/><br/>

## 실행 방법
1. repository를 clone 합니다.
```
git clone https://github.com/junhyeokk/boomhill24.git
```
2. requirements.txt로 필요한 패키지를 설치합니다.
```
pip install -r requirements.txt
```
3. 게임을 실행하고, <code>/demo/driver1_cnn.py</code> 또는 <code>/demo/driver2_cnn_lstm.py</code> 파일의 win_pos 변수 값을 게임 화면의 미니맵에 맞춰줍니다.
```python
self.win_pos = {"top": self.rect[1] + 389, "left": self.rect[0] + 1037, "width": 223, "height": 212}
```
4. <code>/demo/driver1_cnn.py</code> 또는 <code>/demo/driver2_cnn_lstm.py</code>을 실행한 뒤 start 버튼을 누르면 모델의 가중치를 로드한 뒤 추론을 시작합니다.
```
python ./demo/driver1_cnn.py
python ./demo/driver2_cnn_lstm.py
```
<br/>

## 진행 상황
1. CNN 모델<br/>
 CNN 기반의 자율주행을 구현한 <a href="https://arxiv.org/abs/2010.08776">참고 논문</a>의 내용처럼 ResNet 기반 모델을 이용해 0.1초마다 ↑, ←, → 3개 키의 8가지 조합 중 하나를 추론하도록 하였습니다. 속도변화가 적은 18개의 맵에서 3번씩 주행하며 만든 데이터로 학습시켰고, 일반적인 등속운동 코스에서 학습내용과 유사하게 무난하게 동작합니다. 그러나 매 프레임을 독립적으로 받아들이기 때문에 충돌 등으로 인한 급격한 속도변화가 발생하면 대처하지 못하는 문제가 있고, 따라서 부스터, 드리프트 등의 기술을 학습시키기는 어려웠습니다.<br/>
 학습에 사용된 데이터셋의 목록은 <a href="https://docs.google.com/spreadsheets/d/11S4tDy8kQD3ZgfF8kO764H3j7U_tPxEyR7JOc4OrMf8/edit?usp=sharing">링크</a>에서 확인 가능합니다.
 <br/><br/>
<img src="./images/cnn_1.gif">
일반적인 코스에서 무난하게 주행
<br/><br/>
<img src="./images/cnn_2.gif">
충돌로 인한 속도변화 발생시 주행오류
<br/><br/>

2. CNN + LSTM 모델<br/>
 모델이 이전의 상황까지 고려할 수 있도록 CNN과 LSTM을 결합한 형태의 모델로 <a href="https://arxiv.org/abs/2002.05878">참고논문</a>의 모델 설계를 참고하여 0.초마다 ↑, ←, →, ↓, shift, ctrl 6개 키의 64가지 조합 중 하나를 추론하도록 하였습니다. 인터넷 방송의 게임화면과 키뷰어에서 데이터를 추출해 학습시켰지만, 컴퓨팅 자원의 한계로 전체 데이터를 학습시킬만큼 큰 모델을 사용하지 못했습니다. 아래의 이미지는 여러 모델 중 미니맵 입력만을 사용한 것으로 부분적으로 기술을 흉내내지만 전체적인 주행은 불가능한 상태입니다.<br/>
 학습에 사용된 데이터셋의 목록은 <a href="https://docs.google.com/spreadsheets/d/1Augj-bmggBVMYgFhV_q_hS8As6uHnfFNUbngSFL3gNI/edit?usp=sharing">링크</a>에서 확인 가능합니다.
<img src="./images/cnn_lstm_1.gif">
<br/><br/>

3. 시도해볼 것
 * 모델을 학습시키는 시간과 드라이버에서 0.1초마다 추론해야하는 제약으로 전체 데이터를 학습할만한 표현력이 큰 모델을 학습시키지 못하였는데, 추후 기회가 된다면 시도해볼만합니다.
 * 부스터 갯수, 속도, 부스터 게이지 등의 데이터도 화면에서 추출했지만 추론에 사용하지는 못하였는데, 이 데이터들도 모델의 입력으로 추가해볼만 합니다.
 * 프로젝트 초기에 CNN 피쳐와 RNN 계열 모델을 결합한 여러가지 구조를 테스트해보다 중반부터는 2번 모델의 구조를 주로 실험하였는데, 앞선 방법들이나 다른 형태의 모델들을 제대로 만들어 다시 해보면 더 좋은 결과를 얻을 수도 있을것 같습니다.
<br/><br/>

## 관련 링크
* <a href="./demo/initial_plan">초기기획</a>
* <a href="https://docs.google.com/document/d/1stkhaSqZ0RTQ6JEKugAGN491QdeIRplc/edit?usp=sharing&ouid=107588036944538460813&rtpof=true&sd=true">모델 실험 기록</a>
* <a href="https://youtu.be/0fTZ9VjVcr8">데모영상</a>
* <a href="./demo/poster.pdf">포스터</a>
* <a href="./demo/presentation.pptx">발표자료</a>
* <a href="https://youtu.be/145Zlh1SEpE">발표연습</a>