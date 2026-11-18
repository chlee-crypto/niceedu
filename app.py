import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# 페이지 기본 설정
st.set_page_config(layout="wide", page_title="ETF 자산배분 및 레버리지 대시보드")

# ==========================================
# 1. 데이터 로딩 (캐시 적용)
# ==========================================
@st.cache_data
def load_data():
    """CSV 데이터를 로드하고 datetime 변환 후 종가 피벗 테이블을 생성합니다."""
    csv_path = Path(__file__).parent / "data" / "etf_prices.csv"
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    pivot_df = df.pivot(index="datetime", columns="symbol", values="close").sort_index()
    return pivot_df


try:
    prices_data = load_data()
except Exception as e:
    st.error(f"데이터 파일을 불러오는 데 실패했습니다: {e}")
    st.stop()


# ==========================================
# 2. 공통 계산 함수 (두 탭이 공유)
# ==========================================
def calc_total_return(prices):
    """가격 시계열의 총수익률을 계산합니다."""
    return (prices.iloc[-1] / prices.iloc[0]) - 1.0


def calc_cagr(prices):
    """연환산수익률(CAGR)을 계산합니다. 연수 = 경과일수 / 365.25"""
    total_ret = calc_total_return(prices)
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return ((1.0 + total_ret) ** (1.0 / years)) - 1.0


def calc_volatility(prices):
    """일간수익률의 연변동성을 계산합니다. (std * sqrt(252))"""
    daily_ret = prices.pct_change().dropna()
    return daily_ret.std(ddof=1) * np.sqrt(252)


def calc_mdd(prices):
    """고점 대비 최대낙폭(MDD)을 계산합니다."""
    cummax = prices.cummax()
    drawdown = (prices / cummax) - 1.0
    return drawdown.min()


# ==========================================
# 3. 사이드바 (공통 설정)
# ==========================================
st.sidebar.header("설정 및 분석 기간")

min_date = prices_data.index.min().date()
max_date = prices_data.index.max().date()

start_date = st.sidebar.date_input(
    "분석 시작일",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)
end_date = st.sidebar.date_input(
    "분석 종료일",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

# 배당 미포함 주의 문구 상시 고정
st.sidebar.markdown("---")
st.sidebar.warning("주의: 종가 기준 가격 수익률이며 배당은 포함되지 않았습니다. 실제 성과와 다릅니다.")

# 날짜 유효성 검증
if start_date >= end_date:
    st.error("오류: 분석 시작일은 종료일보다 빨라야 합니다. 날짜 설정을 확인해 주세요.")
    st.stop()

# 기간 슬라이싱
sliced_prices = prices_data.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]

if len(sliced_prices) < 60:
    st.sidebar.warning("구간이 짧아 연환산 지표가 불안정합니다.")

# 대시보드 메인 타이틀
st.title("미국 ETF 투자 분석 대시보드")

# 탭 생성
tab1, tab2 = st.tabs(["레버리지의 함정", "분산투자의 힘"])


# ==========================================
# 4. 탭 1 — 레버리지의 함정
# ==========================================
with tab1:
    st.header("레버리지 ETF의 장기 투자 위험성 분석")

    # 대상 종목 고정: QQQ, QLD, TQQQ
    lev_symbols = ["QQQ", "QLD", "TQQQ"]
    lev_prices = sliced_prices[lev_symbols]

    # 3-5. 핵심 메시지 자동 생성
    qqq_ret = calc_total_return(lev_prices["QQQ"])
    tqqq_ret = calc_total_return(lev_prices["TQQQ"])
    tqqq_mdd = calc_mdd(lev_prices["TQQQ"])
    years_elapsed = (sliced_prices.index[-1] - sliced_prices.index[0]).days / 365.25

    if qqq_ret <= 0:
        verdict = f"이 구간은 QQQ(1배) 자체가 {qqq_ret * 100:+.1f}%로 하락하여 배수 비교가 성립하지 않습니다."
    else:
        mult = tqqq_ret / qqq_ret
        if mult < 1.0:
            verdict = f"추종 배수는 3배이지만 실제 수익률은 1배인 QQQ보다도 낮았습니다."
        elif mult < 3.0:
            verdict = f"배수는 3배지만 {years_elapsed:.1f}년 수익률 차이는 {mult:.1f}배에 그쳤습니다."
        else:
            verdict = f"상승 추세가 이어져 수익률 차이가 {mult:.1f}배로 3배를 초과했습니다."

    message_text = (
        f"{start_date} ~ {end_date} 구간에서 TQQQ(3배)는 {tqqq_ret * 100:+.1f}%, QQQ(1배)는 {qqq_ret * 100:+.1f}%였습니다. "
        f"{verdict} 그동안 TQQQ의 최대낙폭은 {tqqq_mdd * 100:.1f}%였습니다."
    )
    st.info(message_text)

    # 3-1 & 3-3. 정규화 성과 곡선 & 낙폭 그래프
    use_log_scale = st.checkbox("로그 스케일 세로축 사용", value=False, key="log_scale_tab1")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("누적 성과 곡선 (기준점 = 100)")
        norm_lev = (lev_prices / lev_prices.iloc[0]) * 100.0
        fig_perf = px.line(
            norm_lev,
            x=norm_lev.index,
            y=lev_symbols,
            labels={"value": "정규화 지수 (100 기준)", "datetime": "날짜"},
            log_y=use_log_scale
        )
        fig_perf.update_layout(legend_title_text="심볼", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_perf, width='stretch')

    with col2:
        st.subheader("고점 대비 낙폭 (Drawdown)")
        dd_df = pd.DataFrame(index=lev_prices.index)
        for sym in lev_symbols:
            cummax = lev_prices[sym].cummax()
            dd_df[sym] = ((lev_prices[sym] / cummax) - 1.0) * 100.0

        fig_dd = px.line(
            dd_df,
            x=dd_df.index,
            y=lev_symbols,
            labels={"value": "낙폭 (%)", "datetime": "날짜"}
        )
        fig_dd.update_layout(legend_title_text="심볼", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_dd, width='stretch')

    # 3-2. 성과 요약표
    st.subheader("선택 기간 성과 요약")
    summary_rows = []
    for sym in lev_symbols:
        tot = calc_total_return(lev_prices[sym])
        cagr = calc_cagr(lev_prices[sym])
        vol = calc_volatility(lev_prices[sym])
        mdd = calc_mdd(lev_prices[sym])
        summary_rows.append({
            "심볼": sym,
            "총수익률": f"{tot * 100:+.1f}%",
            "연환산수익률(CAGR)": f"{cagr * 100:.1f}%",
            "연변동성": f"{vol * 100:.1f}%",
            "최대낙폭(MDD)": f"{mdd * 100:.1f}%"
        })
    st.table(pd.DataFrame(summary_rows))
    st.markdown("""
    - **총수익률**: 선택 기간 동안 자산 가격의 단순 누적 변동률입니다.
    - **연환산수익률(CAGR)**: 기하평균 기준으로 1년 복리 환산한 성장률입니다.
    - **연변동성**: 일간 수익률의 표준편차를 252영업일 기준으로 연환산한 수치입니다.
    - **최대낙폭(MDD)**: 고점 대비 얼마나 빠졌는가를 나타내는 최대 하락 비율입니다.
    """)

    # 3-4. 원금 회복 계산기
    st.subheader("원금 회복 계산기 (하락과 상승의 비대칭성)")
    loss_pct = st.slider("하락률 (%)", min_value=0, max_value=90, value=50, step=5)
    loss_ratio = loss_pct / 100.0
    required_gain = ((1.0 / (1.0 - loss_ratio)) - 1.0) * 100.0 if loss_ratio < 1.0 else float("inf")

    st.metric(label="원금 회복에 필요한 상승률", value=f"{required_gain:+.1f}%")
    st.markdown("-50% 하락은 +100% 상승이 있어야 원금입니다. 하락과 상승은 대칭이 아닙니다.")

    # 3-6. 시작일 변경 유도
    st.markdown("---")
    st.info("사이드바에서 분석 시작일을 2023-01-03으로 바꿔 보십시오. 결론이 뒤집힙니다.")


# ==========================================
# 5. 탭 2 — 분산투자의 힘
# ==========================================
with tab2:
    st.header("자산 분산을 통한 포트폴리오 위험 감소 분석")

    # 4-1. 상관계수 히트맵
    st.subheader("7종목 일간수익률 상관계수 행렬")
    daily_returns = sliced_prices.pct_change().dropna()
    corr_matrix = daily_returns.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        zmin=-1.0,
        zmax=1.0,
        color_continuous_scale="RdBu_r",
        labels=dict(color="상관계수")
    )
    st.plotly_chart(fig_corr, width='stretch')

    # 4-2. 히트맵 해석 문장
    spy_vti = corr_matrix.loc["SPY", "VTI"]
    qqq_tqqq = corr_matrix.loc["QQQ", "TQQQ"]
    gld_spy = corr_matrix.loc["GLD", "SPY"]
    gld_schd = corr_matrix.loc["GLD", "SCHD"]

    st.markdown(f"""
    - **SPY와 VTI의 상관계수 ({spy_vti:.2f})**: 1에 매우 가까워 둘 다 보유하는 것은 실질적인 분산이 아닙니다.
    - **QQQ, QLD, TQQQ (QQQ-TQQQ: {qqq_tqqq:.2f})**: 배수만 다를 뿐 사실상 같은 자산입니다.
    - **GLD의 낮은 상관계수 (SPY: {gld_spy:.2f}, SCHD: {gld_schd_corr if 'gld_schd_corr' in locals() else gld_schd:.2f})**: GLD만 다른 종목들과 상관이 낮으며, 여기서 진정한 분산 효과가 나옵니다.
    """)

    # 4-3. 두 자산 배분 실험 (SPY & GLD)
    st.subheader("두 자산 배분 실험 (SPY & GLD)")
    st.caption("가정: 매일 비중을 다시 맞춘다는 일간 리밸런싱(Daily Rebalancing) 기준입니다.")

    spy_weight = st.slider("SPY 비중 (%)", min_value=0, max_value=100, value=50, step=5, key="spy_slider")
    gld_weight = 100 - spy_weight

    w_spy = spy_weight / 100.0
    w_gld = gld_weight / 100.0

    port_daily_ret = (w_spy * daily_returns["SPY"]) + (w_gld * daily_returns["GLD"])
    port_prices = pd.Series(100.0, index=sliced_prices.index)
    port_prices.iloc[1:] = 100.0 * (1.0 + port_daily_ret).cumprod()

    # 지표 계산
    port_tot = calc_total_return(port_prices)
    port_vol = calc_volatility(port_prices)
    port_mdd = calc_mdd(port_prices)

    # SPY 100% 대조군
    spy_only_prices = sliced_prices["SPY"]
    spy_tot = calc_total_return(spy_only_prices)
    spy_vol = calc_volatility(spy_only_prices)
    spy_mdd = calc_mdd(spy_only_prices)

    diff_tot = port_tot - spy_tot
    diff_vol = port_vol - spy_vol
    diff_mdd = port_mdd - spy_mdd

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            "포트폴리오 총수익률",
            f"{port_tot * 100:.1f}%",
            delta=f"{diff_tot * 100:+.1f}%p (vs SPY 100%)"
        )
    with m2:
        st.metric(
            "포트폴리오 연변동성",
            f"{port_vol * 100:.1f}%",
            delta=f"{diff_vol * 100:+.1f}%p (vs SPY 100%)",
            delta_color="inverse"
        )
    with m3:
        st.metric(
            "포트폴리오 최대낙폭(MDD)",
            f"{port_mdd * 100:.1f}%",
            delta=f"{diff_mdd * 100:+.1f}%p (vs SPY 100%)",
            delta_color="off"
        )

    # 4-4. 비중별 위험-수익 산점도
    st.subheader("비중별 위험-수익 산점도")
    scatter_list = []
    for w in range(0, 101, 10):
        ws = w / 100.0
        wg = 1.0 - ws
        temp_ret = (ws * daily_returns["SPY"]) + (wg * daily_returns["GLD"])
        temp_prices = pd.Series(100.0, index=sliced_prices.index)
        temp_prices.iloc[1:] = 100.0 * (1.0 + temp_ret).cumprod()

        tot = calc_total_return(temp_prices)
        vol = calc_volatility(temp_prices)

        scatter_list.append({
            "SPY_weight": w,
            "총수익률": tot * 100.0,
            "연변동성": vol * 100.0,
            "label": f"SPY {w}% / GLD {100-w}%"
        })

    scatter_df = pd.DataFrame(scatter_list)

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=scatter_df["연변동성"],
        y=scatter_df["총수익률"],
        mode="markers+text",
        text=scatter_df["SPY_weight"].apply(lambda w: f"SPY {w}%"),
        textposition="top center",
        marker=dict(size=10, color="blue", opacity=0.6),
        name="비중별 조합"
    ))
    fig_scatter.add_trace(go.Scatter(
        x=[port_vol * 100.0],
        y=[port_tot * 100.0],
        mode="markers",
        marker=dict(size=15, color="red", line=dict(width=2, color="black")),
        name="현재 선택 비중"
    ))
    fig_scatter.update_layout(
        xaxis_title="연변동성 (%)",
        yaxis_title="총수익률 (%)",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_scatter, width='stretch')

    min_vol_row = scatter_df.loc[scatter_df["연변동성"].idxmin()]
    gld_only = scatter_df.iloc[0]
    spy_only = scatter_df.iloc[-1]

    st.markdown(f"""
    이 그래프의 왼쪽 끝(최저 변동성 지점)은 **100% 주식(SPY {spy_only['연변동성']:.1f}%)도, 100% 금(GLD {gld_only['연변동성']:.1f}%)도 아닌 SPY {min_vol_row['SPY_weight']:.0f}% / GLD {100 - min_vol_row['SPY_weight']:.0f}% 조합(연변동성 {min_vol_row['연변동성']:.1f}%)**입니다.
    서로 상관관계가 낮은 자산을 섞음으로써 개별 자산보다 전체 포트폴리오의 변동성이 오히려 낮아지는 분산투자의 효과가 실증됩니다.
    """)

    # 4-5. 자산군 성과 비교 (SPY, GLD, SCHD)
    st.subheader("주요 자산군 누적 성과 비교 (SPY, GLD, SCHD)")
    comp_symbols = ["SPY", "GLD", "SCHD"]
    comp_prices = sliced_prices[comp_symbols]
    norm_comp = (comp_prices / comp_prices.iloc[0]) * 100.0

    fig_comp = px.line(
        norm_comp,
        x=norm_comp.index,
        y=comp_symbols,
        labels={"value": "정규화 지수 (100 기준)", "datetime": "날짜"}
    )
    fig_comp.update_layout(legend_title_text="자산군", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_comp, width='stretch')
    st.caption("주의: SCHD는 배당이 반영되지 않은 가격 수익률 기준이므로 실제 분배금을 포함한 누적 성과 대비 과소평가되어 있습니다.")
