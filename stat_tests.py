import pandas as pd
import numpy as np
from scipy.stats import (ttest_ind,
                         ttest_rel)
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

ling_data = pd.read_csv("features/language_features.csv")
ling_data.columns

sents_data = pd.read_csv("features/sentiment_features.csv")
sents_data.columns

def cal_stats_ind(sample1, sample2):
    t_stat, p_value = ttest_ind(sample1, sample2, equal_var=False, nan_policy='omit')
    cohen_d = (np.mean(sample1) - np.mean(sample2)) / np.sqrt((np.std(sample1) ** 2 + np.std(sample2) ** 2) / 2)
    return t_stat, p_value, cohen_d

def cal_stats_paired(sample1, sample2):
    t_stat, p_value = ttest_rel(sample1, sample2, nan_policy='omit')
    cohen_d = (np.mean(sample1) - np.mean(sample2)) / np.sqrt((np.std(sample1) ** 2 + np.std(sample2) ** 2) / 2)
    return t_stat, p_value, cohen_d

# ----------------------- Language --------------------------

# --- Word Length ---
# real vs. fake
t_stat_wl_rf, p_value_wl_rf, cohen_d_wl_rf = cal_stats_ind(ling_data['real_word_len_list'], ling_data['fake_word_len_list'])

# real vs. ai
t_stat_wl_ra, p_value_wl_ra, cohen_d_wl_rf = cal_stats_ind(ling_data['real_word_len_list'], ling_data['ai_word_len_list'])

# --- Sentence Length ---
# real vs. fake
t_stat_sl_rf, p_value_sl_rf, cohen_d_sl_rf = cal_stats_ind(ling_data['real_sentence_len_list'], ling_data['fake_sentence_len_list'])

# real vs. ai
t_stat_sl_ra, p_value_sl_ra, cohen_d_sl_ra = cal_stats_ind(ling_data['real_sentence_len_list'], ling_data['ai_sentence_len_list'])

# --- Article Length (sentences) ---
# real vs. fake
t_stat_al_rf, p_value_al_rf, cohen_d_al_rf = cal_stats_ind(ling_data['real_article_len_list'], ling_data['fake_article_len_list'])

# real vs. ai
t_stat_al_ra, p_value_al_ra, cohen_d_al_ra = cal_stats_ind(ling_data['real_article_len_list'], ling_data['ai_article_len_list'])

# --- Article Length (tokens) ---
# real vs. fake
t_stat_alt_rf, p_value_alt_rf, cohen_d_alt_rf = cal_stats_ind(ling_data['real_no_word_list'], ling_data['fake_no_word_list'])

# real vs. ai
t_stat_alt_ra, p_value_alt_ra, cohen_d_alt_ra = cal_stats_paired(ling_data['real_no_word_list'], ling_data['ai_no_word_list'])

# --- No of Verifiable facts ---
# real vs. fake
t_stat_vf_rf, p_value_vf_rf, cohen_d_vf_rf = cal_stats_ind(ling_data['real_no_verifiable_facts'], ling_data['fake_no_verifiable_facts'])

# real vs. ai
t_stat_vf_ra, p_value_vf_ra, cohen_d_vf_ra = cal_stats_paired(ling_data['real_no_verifiable_facts'], ling_data['ai_no_verifiable_facts'])

# --- Percentage of Verifiable facts ---
# real vs. fake
t_stat_per_vf_rf, p_value_per_vf_rf, cohen_d_per_vf_rf = cal_stats_ind(ling_data['real_percentage_verifiable_facts'], ling_data['fake_percentage_verifiable_facts'])

# real vs. ai
t_stat_per_vf_ra, p_value_per_vf_ra, cohen_d_per_vf_ra = cal_stats_paired(ling_data['real_percentage_verifiable_facts'], ling_data['ai_percentage_verifiable_facts'])

# ----------------------- Sentiments --------------------------

# --- Polarity ---
# real vs. fake
t_stat_pol_rf, p_value_pol_rf, cohen_d_pol_rf = cal_stats_ind(sents_data['real_article_pol'], sents_data['fake_article_pol'])

# real vs. ai
t_stat_pol_ra, p_value_pol_ra, cohen_d_pol_ra = cal_stats_paired(sents_data['real_article_pol'], sents_data['ai_article_pol'])

# --- Sentiment Volatility ---
# real vs. fake
t_stat_sv_rf, p_value_sv_rf, cohen_d_sv_rf = cal_stats_ind(sents_data['real_sv'], sents_data['fake_sv'])

# real vs. ai
t_stat_sv_ra, p_value_sv_ra, cohen_d_sv_ra = cal_stats_paired(sents_data['real_sv'], sents_data['ai_sv'])

# --- No. of affect words ---
# real vs. fake
t_stat_affect_rf, p_value_affect_rf, cohen_d_affect_rf = cal_stats_ind(sents_data['real_no_affect_word'], sents_data['fake_no_affect_word'])

# real vs. ai
t_stat_affect_ra, p_value_affect_ra, cohen_d_affect_ra = cal_stats_paired(sents_data['real_no_affect_word'], sents_data['ai_no_affect_word'])

# --- Percentage of affect words ---
# real vs. fake
t_stat_per_affect_rf, p_value_per_affect_rf, cohen_d_per_affect_rf = cal_stats_ind(sents_data['real_percentage_affect_words'], sents_data['fake_percentage_affect_words'])

# real vs. ai
t_stat_per_affect_ra, p_value_per_affect_ra, cohen_d_per_affect_ra = cal_stats_paired(sents_data['real_percentage_affect_words'], sents_data['ai_percentage_affect_words'])

# --- Subjectivity ---
# real vs. fake
t_stat_sub_rf, p_value_sub_rf, cohen_d_sub_rf = cal_stats_ind(sents_data['real_article_sub'], sents_data['fake_article_sub'])

# real vs. ai
t_stat_sub_ra, p_value_sub_ra, cohen_d_sub_ra = cal_stats_paired(sents_data['real_article_sub'], sents_data['ai_article_sub'])