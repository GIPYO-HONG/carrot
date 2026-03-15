import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
from datetime import datetime

# 기존 라이브러리 임포트
from .utiles import *

########## model define ##########

class PINN(eqx.Module):
    S_net: eqx.nn.MLP
    E_net: eqx.nn.MLP
    I_net: eqx.nn.MLP
    A_net: eqx.nn.MLP
    R_net: eqx.nn.MLP
    bb_net: eqx.nn.MLP

    def __init__(self, width_size, depth, activation, final_activation, *, key):
        keys = jr.split(key, 6)
        def make_mlp(k):
            return eqx.nn.MLP(
                in_size=1, out_size=1,
                width_size=width_size, depth=depth,
                activation=activation, final_activation=final_activation,
                key=k
            )
        self.S_net, self.E_net, self.I_net, self.A_net, self.R_net, self.bb_net = [make_mlp(k) for k in keys]

    def __call__(self, t):
        # t: (1,) shape array
        return jnp.array([
            self.S_net(t)[0], self.E_net(t)[0], self.I_net(t)[0],
            self.A_net(t)[0], self.R_net(t)[0], self.bb_net(t)[0]
        ])

########## Experiment ##########

class Experiment(BaseExperiment):
    def __init__(self, beta, y0, ts, ys, ts_ge=jnp.linspace(0., 365., 365*2+1), I_only=False, width_size=64, depth=2, **kwargs):
        seed = kwargs.get('seed', 5678)
        # PINN 모델 생성
        model = PINN(
            width_size=width_size, 
            depth=depth, 
            activation=lambda x: jnn.tanh(x), 
            final_activation=lambda x: jnn.sigmoid(x), 
            key=jr.PRNGKey(seed)
        )
        
        super().__init__(model, y0, ts, ys, **kwargs)
        
        self.I_only = I_only
        self.beta = beta
        self.ts_ge = ts_ge # 물리 법칙을 강제할 콜로케이션 포인트
        self.scales = jnp.max(ys, axis=0) + 1e-4

    def loss_fn(self, model, ts, ys):
        # 1. Data Loss: 관측 데이터와 모델 출력의 차이
        # t를 넣었을 때 [S, E, I, A, R, beta]가 나옴
        preds = jax.vmap(model)(ts[:, None]) 
        y_pred = preds[:, :5]
        
        if self.I_only:
            data_loss = jnp.mean(jnp.square((y_pred[:, 2] - ys[:, 2]) / self.scales[2]))
        else:
            data_loss = jnp.mean(jnp.square((y_pred - ys) / self.scales))

        # 2. Physics Loss (Residuals): ODE를 만족하는지 검사
        # ts_ge 포인트에서 자동 미분(Jacobian)을 사용하여 미분값 계산
        def get_residuals(t_point):
            # t_point에서의 상태값과 시간 미분값 계산
            y_bb, dydt = jax.jvp(model, (t_point,), (jnp.ones_like(t_point),))
            y_s = y_bb[:5]
            bb_s = y_bb[5]
            
            # SEIAR 물리 방정식의 우변(RHS) 계산
            rhs = SEIAR(0, y_s, bb_s)
            return dydt[:5] - rhs # Residual = (LHS - RHS)

        res = jax.vmap(get_residuals)(self.ts_ge[:, None])
        physics_loss = jnp.mean(jnp.square(res / self.scales))

        total_loss = data_loss + physics_loss
        return total_loss, None
    
########## Evaluation ##########

def Evaluation(EX, ts_eval, loss_list, viz_data=False):
    """
    PINN 실험 결과를 평가하고 시각화합니다.
    """
    y0, ts_data, ys_data, model, I_only = EX.y0, EX.ts_data, EX.ys_data, EX.model, EX.I_only

    # 1. Ground Truth 데이터 생성 (참값)
    # get_data가 (ts, 6) 형태(S,E,I,A,R,beta)를 반환한다고 가정
    gt_all = get_data(ts_eval, y0, EX.beta) 
    ys_eval = gt_all[:, :5]
    beta_eval = EX.beta(ts_eval)

    # 2. PINN 모델 예측 (vmap 사용)
    # PINN 모델은 t를 넣으면 [S, E, I, A, R, bb]를 반환함
    preds = jax.vmap(model)(ts_eval[:, None])
    ys_pred = preds[:, :5]  # SEIAR 예측값
    beta_pred = preds[:, 5] # Beta 예측값

    # 3. 기존 공통 plotting 함수 호출
    # 기존 plotting 함수가 (ts_data, ys_data, ts_eval, ys_eval, ys_pred, beta_eval, beta_pred, ...) 
    # 인자 순서를 가진다고 가정했을 때:
    plotting(
        ts_data, 
        ys_data, 
        ts_eval, 
        ys_eval, 
        ys_pred, 
        beta_eval, 
        beta_pred, 
        loss_list, 
        I_only, 
        viz_data
    )


########## Test Execution ##########

if __name__ == '__main__':
    exp_name = datetime.now().strftime("exp_PINN_%Y%m%d_%H%M%S")

    # 데이터 생성
    beta_obj = beta_generate(5e-1, 0.1, 0.)
    y0 = jnp.array([1e+0, 0., 1e-6, 0., 0.])
    ts = jnp.linspace(0., 365., 100) # 관측 데이터는 100개만 있다고 가정
    ys = get_data(ts, y0, beta_obj.func)

    # PINN을 위한 격자점 (Collocation points) - 더 촘촘하게 설정
    ts_ge = jnp.linspace(0., 365., 500)

    EX = Experiment(
        beta=beta_obj.func, 
        y0=y0, 
        ts=ts, 
        ys=ys, 
        ts_ge=ts_ge,
        I_only=False, 
        exp_name=exp_name
    )

    # 학습 (PINN은 ODE Solver를 안 쓰기 때문에 step당 속도는 빠르지만 더 많은 step이 필요할 수 있음)
    EX.train(lr=1e-3, steps=20000, print_every=2000)