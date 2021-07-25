
# resources ---------------------------------------------------------------
# Kaggle competition: https://www.kaggle.com/c/tabular-playground-series-jul-2021/overview

# objective ---------------------------------------------------------------
#' - submit our predictions to the Kaggle July TPS competition
#'   -- apply SLICED methods:
#'      --- bagged tree model
#'      --- random forest model
#'      --- xgboost (see Dave Robinson's code)


# setup -------------------------------------------------------------------
library(tidymodels)
library(tidyverse)
library(baguette)
library(DataExplorer)
library(janitor)
library(here)

tidymodels_prefer()
options(tidymodels.dark = TRUE)

grid_control <- control_grid(save_pred = TRUE,
                             save_workflow = TRUE,
                             verbose = TRUE,
                             extract = extract_model)


# import our data ---------------------------------------------------------
path <- "00_2021-07-25_kaggle_july_tps/data"

train <- read_csv(here(path, "train.csv"))
test <- read_csv(here(path, "test.csv"))
sampsub <- read_csv(here(path, "sample_submission.csv"))

glimpse(train)
glimpse(test)
glimpse(sampsub)

compare_df_cols(train, test)

plot_intro(train)
plot_histogram(train)
# (c(target_1, target_2, target_3) ~ ., )


# spend our data ----------------------------------------------------------
set.seed(406)
folds <- vfold_cv(v = 5, data = train)


# three bagged trees lol --------------------------------------------------
# carbon monoxide
bb_bt_co <- recipe(target_carbon_monoxide ~ date_time, deg_C, relative_humidity,
                   sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, 
                   data = train) %>% 
  update_role(date_time, new_role = "id")
glimpse(bb_bt_co)

bb_bt_spec <- bag_tree(min_n = 10) %>%
  set_engine('rpart', times = 25) %>%
  set_mode('regression')

bb_bt_wflow_co <- workflow() %>%
  add_recipe(bb_bt_co) %>%
  add_model(bb_bt_spec)

bb_bt_res_co <- fit_resamples(
  bb_bt_wflow_co,
  folds,
  control = grid_control
)

collect_metrics(bb_bt_res_co)

bb_bt_fit_co <- fit(bb_bt_wflow_co, train)
bb_bt_pred_co <- augment(bb_bt_fit_co, new_data = test)  

glimpse(bb_bt_pred_co) 

# benzene
bb_bt_bnz <- recipe(target_benzene ~ date_time, deg_C, relative_humidity,
                   sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, 
                   data = train) 

bb_bt_wflow_bnz <- workflow() %>%
  add_recipe(bb_bt_bnz) %>%
  add_model(bb_bt_spec)

bb_bt_res_bnz <- fit_resamples(
  bb_bt_wflow_bnz,
  folds,
  control = grid_control
)

collect_metrics(bb_bt_res_bnz)

bb_bt_fit_bnz <- fit(bb_bt_wflow_bnz, train)
bb_bt_pred_bnz <- augment(bb_bt_fit_bnz, new_data = test)  

glimpse(bb_bt_pred_bnz) 

# nitrogen oxides
bb_bt_no <- recipe(target_nitrogen_oxides ~ date_time, deg_C,
                    relative_humidity,
                    sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, 
                    data = train) 

bb_bt_wflow_no <- workflow() %>%
  add_recipe(bb_bt_no) %>%
  add_model(bb_bt_spec)

bb_bt_res_no <- fit_resamples(
  bb_bt_wflow_no,
  folds,
  control = grid_control
)

collect_metrics(bb_bt_res_no)

bb_bt_fit_no <- fit(bb_bt_wflow_no, train)
bb_bt_pred_no <- augment(bb_bt_fit_no, new_data = test)  

glimpse(bb_bt_pred_no)

s# bind columns for final submission
bt_co <- bb_bt_pred_co %>% 
  # CHECK FIRST
  select(date_time,
         target_carbon_monoxide = .pred) 

bt_bnz <- bb_bt_pred_bnz %>% 
  select(date_time, 
         target_benzene = .pred)

bt_no <- bb_bt_pred_no %>% 
  select(date_time,
         target_nitrogen_oxides = .pred)

bt_co %>% 
  left_join(bt_bnz, by = "date_time") %>% 
  left_join(bt_no, by = "date_time") %>% 
  write_csv(here(path, "bb_bt_round_01.csv"))

bt_sub <- read_csv(here(path, "bb_bt_round_01.csv"))
glimpse(bt_sub)

# notes -------------------------------------------------------------------
#' - with the {tidymodels} framework, we may not be able to do multivariate
#'   predictions. in other words, we'll predict each of the outcomes separately
#'   and then combine them into a .csv for submission
#'   -- ideally we'd build a model that simultaneously predicts all three of 
#'      our response outcome
#'      
#' - may want to log transform the outcome variables -- each is skewed
