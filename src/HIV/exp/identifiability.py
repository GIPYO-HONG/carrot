from juliacall import Main as jl

# 1. 라이브러리 로드
jl.seval("using StructuralIdentifiability")

print("--- [경량화] 시간 가변 beta(t) 로컬 식별성 분석 시작 ---")

# 2. 모델 정의 및 고속 분석
# assess_local_identifiability를 사용하여 계산량을 대폭 줄입니다.
results = jl.seval("""
    ODE = @ODEmodel(
        S'(t) = -beta(t) * S(t) * (0.5 * I(t) + A(t)),
        E'(t) = beta(t) * S(t) * (0.5 * I(t) + A(t)) - 0.526 * E(t),
        I'(t) = 0.667 * 0.526 * E(t) - 0.244 * I(t),
        A'(t) = (1 - 0.667) * 0.526 * E(t) - 0.244 * A(t),
        beta'(t) = u(t), 
        y(t) = I(t)
    )
    
    # 전역(Global) 대신 로컬(Local) 식별성만 체크 (훨씬 빠름)
    # 결과는 OrderedDict{Any, Bool} 형태로 나옵니다.
    assess_local_identifiability(ODE)
""")

print("\n--- 분석 결과 (True면 식별 가능) ---")
print(results)