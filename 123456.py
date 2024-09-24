import numpy as np

class Activation_Step:
    def forward(self, inputs):
        # 입력이 0보다 크면 1, 그렇지 않으면 0으로 변환
        return (inputs > 0).astype(float)

# X 값 생성
X = np.linspace(0.2 * np.pi, 100).reshape(-1, 1)

# y = sin(X) 값 생성
y = np.sin(X)

# Activation_Step 클래스 인스턴스화
activation_step = Activation_Step()

# y 값에 대해 활성화 함수 적용
activation_output = activation_step.forward(y)

# 출력
print(activation_output)