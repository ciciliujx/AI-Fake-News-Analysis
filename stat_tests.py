import pandas as pd
from scipy.stats import (ttest_ind,
                         ttest_rel)

data = pd.read_csv("sentiment_features.csv")
data.columns

def cal_stats_real_fake(real, fake):
    t_stat, p_value = ttest_ind(real, fake, equal_var=False)
    return t_stat, p_value

def cal_stats_real_ai(real, ai):
    t_stat, p_value = ttest_rel(real, ai)
    return t_stat, p_value

# ----------------------- Sentiments --------------------------

# --- Polarity ---
# real vs. fake
t_stat_pol_rf, p_value_pol_rf = cal_stats_real_fake(data['real_article_pol'], data['fake_article_pol'])

# real vs. ai
t_stat_pol_ra, p_value_pol_ra = cal_stats_real_ai(data['real_article_pol'], data['ai_article_pol'])

# --- Sentiment Volatility ---
# real vs. fake
t_stat_sv_rf, p_value_sv_rf = cal_stats_real_fake(data['real_sv'], data['fake_sv'])

# real vs. ai
t_stat_sv_ra, p_value_sv_ra = cal_stats_real_ai(data['real_sv'], data['ai_sv'])

# --- No. of affect words ---
# real vs. fake
t_stat_affect_rf, p_value_affect_rf = cal_stats_real_fake(data['real_no_affect_word'], data['fake_no_affect_word'])

# real vs. ai
t_stat_affect_ra, p_value_affect_ra = cal_stats_real_ai(data['real_no_affect_word'], data['ai_no_affect_word'])

# --- Subjectivity ---
# real vs. fake
t_stat_sub_rf, p_value_sub_rf = cal_stats_real_fake(data['real_article_sub'], data['fake_article_sub'])

# real vs. ai
t_stat_sub_ra, p_value_sub_ra = cal_stats_real_ai(data['real_article_sub'], data['ai_article_sub'])