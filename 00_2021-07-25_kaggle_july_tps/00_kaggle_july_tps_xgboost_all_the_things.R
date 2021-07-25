
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
                   data = train) 

#glimpse(bb_bt_co)

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

# bind columns for final submission
bt_co <- bb_bt_pred_co %>% 
  # CHECK FIRST
  select(date_time,
         target_carbon_monoxide = .pred) 
glimpse(bt_co)

bt_bnz <- bb_bt_pred_bnz %>% 
  select(date_time, 
         target_benzene = .pred)
glimpse(bt_bnz)

bt_no <- bb_bt_pred_no %>% 
  select(date_time,
         target_nitrogen_oxides = .pred)
glimpse(bt_no)

bt_co %>% 
  left_join(bt_bnz, by = "date_time") %>% 
  left_join(bt_no, by = "date_time") %>%
  mutate(date_time = strftime(date_time, tz = "UTC")) %>% 
  write_csv(here(path, "bb_bt_round_04.csv"))

bt_sub <- read_csv(here(path, "bb_bt_round_04.csv"))
glimpse(bt_sub)


# three xgboost models because why not ------------------------------------

# # log transformed data --------------------------------------------------
train_log <- train %>% 
  mutate(target_carbon_monoxide = log(target_carbon_monoxide + 1),
         target_benzene = log(target_benzene + 1),
         target_nitrogen_oxides = log(target_nitrogen_oxides + 1))

set.seed(406)
split <- initial_split(train_log)
train_split_co <- training(split)
test_split_co <- testing(split)

set.seed(406)
folds <- vfold_cv(train_log, v = 5)

xgb_rec_co <- recipe(target_carbon_monoxide ~ date_time + deg_C + relative_humidity +
                       sensor_1 + sensor_2 + sensor_3 + sensor_4 + sensor_5, 
                     data = train_log) %>% 
  step_date(date_time, keep_original_cols = FALSE) %>% 
  #update_role(date_time, new_role = "id") %>% 
  step_dummy(all_nominal_predictors())

xgb_rec_co %>% 
  prep() %>% 
  juice() %>% 
  glimpse()

xgb_wf_co <- workflow() %>% 
  add_recipe(xgb_rec_co) %>% 
  add_model(boost_tree("regression",
                       mtry = tune(),
                       trees = tune(),
                       learn_rate = 0.02) %>% 
              set_engine("xgboost"))

xgb_tune_co <- xgb_wf_co %>% 
  tune_grid(folds,
            grid = crossing(mtry = seq(2, 4),
                            trees = seq(300, 800, 100)),
            control = grid_control)

autoplot(xgb_tune_co)

# from Tony
xgb_tune_co$.notes[[1]]
# from Emil
fit(xgb_wf_co, train_log)

xgb_wf_best_co <- xgb_wf_co %>% 
  finalize_workflow(select_best(xgb_tune_co))

xgb_fit_best_co <- xgb_wf_best_co %>% 
  fit(train_log)

importances <- xgboost::xgb.importance(model = xgb_fit_best_co$fit$fit$fit)

importances %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>%
  ggplot(aes(Gain, Feature)) +
  geom_point()

predictions_co <- xgb_fit_best_co %>% 
  augment(test)

glimpse(predictions_co)

# benzene
xgb_rec_bnz <- recipe(target_benzene ~ date_time + deg_C + relative_humidity +
                       sensor_1 + sensor_2 + sensor_3 + sensor_4 + sensor_5, 
                     data = train_log) %>% 
  step_date(date_time, keep_original_cols = FALSE) %>% 
  #update_role(date_time, new_role = "id") %>% 
  step_dummy(all_nominal_predictors())

xgb_wf_bnz <- workflow() %>% 
  add_recipe(xgb_rec_bnz) %>% 
  add_model(boost_tree("regression",
                       mtry = tune(),
                       trees = tune(),
                       learn_rate = 0.02) %>% 
              set_engine("xgboost"))

xgb_tune_bnz <- xgb_wf_bnz %>% 
  tune_grid(folds,
            grid = crossing(mtry = seq(2, 4),
                            trees = seq(300, 800, 100)),
            control = grid_control)

xgb_wf_best_bnz <- xgb_wf_bnz %>% 
  finalize_workflow(select_best(xgb_tune_bnz))

xgb_fit_best_bnz <- xgb_wf_best_bnz %>% 
  fit(train_log)

predictions_bnz <- xgb_fit_best_bnz %>% 
  augment(test)

glimpse(predictions_bnz)

# nitrogen oxides
xgb_rec_no <- recipe(target_nitrogen_oxides ~ date_time + deg_C + relative_humidity +
                        sensor_1 + sensor_2 + sensor_3 + sensor_4 + sensor_5, 
                      data = train_log) %>% 
  step_date(date_time, keep_original_cols = FALSE) %>% 
  #update_role(date_time, new_role = "id") %>% 
  step_dummy(all_nominal_predictors())

xgb_wf_no <- workflow() %>% 
  add_recipe(xgb_rec_no) %>% 
  add_model(boost_tree("regression",
                       mtry = tune(),
                       trees = tune(),
                       learn_rate = 0.02) %>% 
              set_engine("xgboost"))

xgb_tune_no <- xgb_wf_no %>% 
  tune_grid(folds,
            grid = crossing(mtry = seq(2, 4),
                            trees = seq(300, 800, 100)),
            control = grid_control)

xgb_wf_best_no <- xgb_wf_no %>% 
  finalize_workflow(select_best(xgb_tune_no))

xgb_fit_best_no <- xgb_wf_best_no %>% 
  fit(train_log)

predictions_no <- xgb_fit_best_no %>% 
  augment(test)

glimpse(predictions_no)

# putting it all together
pred_co <- predictions_co %>% 
  mutate(target_carbon_monoxide = expm1(.pred))

pred_bnz <- predictions_bnz %>% 
  mutate(target_benzene = expm1(.pred)) %>% 
  select(date_time, target_benzene)

pred_no <- predictions_no %>% 
  mutate(target_nitrogen_oxides = expm1(.pred)) %>% 
  select(date_time, target_nitrogen_oxides)

xgb_preds <- pred_co %>% 
  select(date_time, target_carbon_monoxide) %>% 
  left_join(pred_bnz, by = "date_time") %>% 
  left_join(pred_no, by = "date_time")

glimpse(xgb_preds)

xgb_preds %>% 
  mutate(date_time = strftime(date_time, tz = "UTC")) %>%
  write_csv(here(path, "xgb_02.csv"))

# notes -------------------------------------------------------------------
#' - with the {tidymodels} framework, we may not be able to do multivariate
#'   predictions. in other words, we'll predict each of the outcomes separately
#'   and then combine them into a .csv for submission
#'   -- ideally we'd build a model that simultaneously predicts all three of 
#'      our response outcome
#'      
#' - may want to log transform the outcome variables -- each is skewed
#' 
#' - xgboost can only handle numerics -- this is important! because you forget 
#'   it! A LOT!
#'   
#' - step_date is one of the few steps that doesn't remove original columns
