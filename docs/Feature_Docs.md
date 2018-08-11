# Model Data Input

## Training-set

Train data gives policy's next year premium. Column names speak for themselves.

## Policy Data

Policy data gives contract level information. One policy is composed of one or more contracts. Each contract differs by insurance coverage, e.g. 16G, 16P, &amp; etc. Part of the column explanation can be found below:

- Insured Amount: Insured amounts are comparable only between contracts of one insurance coverage. Insured amounts are named implicitly within each insurance coverage, but differently across coverages.
- Coverage Deductible If Applied: Same as above.
- Replacement Cost of Insured Vehicle: Replacement cost is in the unit of 10,000. This value is used in premium when the cost of recovering equals to replacement, e.g. theft.
- Distribution Channel: Distribution channels can be categorized into two types, direct and indirect. Direct channel means a multiplier of 1/0.7927 in the premium; indirect channel means a multiplier of 1/0.65 in the premium.

## Claim Data

Claim data has not been examined fully yet.

# Data Preprocess

## Method 1: Group by Main Coverage

This method will group Policy Data and Claim Data at policy number level. Independent variable dimensions will be consistent across policies.

On Policy Data, we aggregated contract level information1 on main coverage groups, and unstacked the information to columns. Steps are as following:

- Policy Data is split into policy level and contract level information.
- On policy level, we can select any row, specifically the first row, for processed independent variable.
- On contract level, we add 'Main_Insurance_Coverage_Group' to index, and aggregate the other 6 columns by:
  - Sum: 'Insured_Amount1', 'Insured_Amount2', 'Insured_Amount3', 'Premium'
  - Count: 'Insurance_Coverage'
  - Binary on Existence: 'Coverage_Deductible_if_applied'
- On contract level, unstack index 'Main_Insurance_Coverage_Group' into 18 columns. Contract level information is aggregated at policy level now.

On Claim Data, we aggregated claim level information2 on main coverage groups, and unstacked the information to columns. Steps are as following:

- On claim level, we add 'Main_Insurance_Coverage_Group' to index, and aggregate the other 7 columns by:
  - Sum: 'Paid_Loss_Amount', 'paid_Expenses_Amount', 'Salvage_or_Subrogation?', 'Deductible', 'number_of_claimants'
  - Count: 'Claim_Number'
  - Mean: 'At_Fault?'
- On claim level, unstack index 'Main_Insurance_Coverage_Group' into 21 columns. Contract level information is aggregated at policy level now.

## Method2: Group by ID

This method aggregates by policy number on dimensions including insurer, vehicle branding, coverage, and claims.

On Insurer level, following features will be included:

- Insured's_ID: significance of this column indicates the potential of feature engineering on the insured.
- fsex
- fmarriage
- iage_lab: separate by 25, 30, and 60 above.
- iassured_lab: separate Legal from natural person. (info lost)
- iclaims: number of claims from the insured
- iclaim_paid_amount: total amount of expenses and loss paid on accidents by insured
- iclaim_salvage_amount: total amount retrieved from salvage and subrogation
- ipolicies: number of policies held by the insured
- ipolicy_coverage_avg: average number of coverage of each policy
- ipolicy_premium_avg: average premium paid by the insured
- ivehicle_repcost_avg: average replacement cost.
- (ipolicy_premium_chg: average change of premium)

On Vehicle level, following features will be included:

- Vehicle_identifier: significance of this column indicates the potential of feature engineering on the vehicle.
- Coding_of_Vehicle_Branding_&amp;_Type
- vyear_lab: separate by 0y, 1y, 2y, 3y, and above every 5 year.
- vregion_lab: separate by domestic and imported.
- vengine_lab: group engine displacement by 1000 cubic centenmeter
- vlocomotive: separate locomotive from others
- Vehicle_Make_and_Model1
- Vehicle_Make_and_Model2

On Main Coverage level, following features will be included:

- cpremium_dmg: sum of premium on damage contracts
- cpremium_lia: sum of premium on liability contracts
- cpremium_thf: sum of premium on theft contracts
- cbucket: bucket ID for coverage written in policy

On Coverage level, following features will be indlcuded:

- cpremium_dmg_acc_adj: adjust according to pdmg_acc, sex and age
- cpremium_lia_acc_adj: adjust according to plia_acc
- (cinsured_scaled: weighted average of insured amount rank within each coverage)
- (cdistr_multi: cost of distribution can be either 35% or 29.27%)
- (Premium_adj: adjusted premium according to number of claims received by the insurer/policy)

On Policy Data level, following features will be included

- aassured_zip
- iply_area
- Distribution_Channel
- Multiple_Products_with_TmNewa_(Yes_or_No?)

On Claim Data level, following features will be included

- cclaim_id: significance of this column indicates the potential of feature engineering on the claim.
- cclaims: number of claims on the policy
- closs: sum of loss and expenses on the policy
- csalvate: sum of salvage and surbrogation on the policy
- ccause_type: bucket of accident causes

## Method3: Baseline Dataet

This dataset should involve minimal minipulation while keep the most amount of information. This model will be the baseline for dimension reduction, feature engineering and feature seclection. Following are columns to be involved:

- cat_age: group 5 years into one label; e.g. cat_age = 5 for ibirth = 07/1988
- cat_area: iply_area
- cat_assured: fassured
- cat_cancel: Cancellation
- cat_distr: Distribution_Channel
- cat_marriage: fmarriage
- cat_sex: fsex
- cat_vc: Coding_of_Vehicle_Branding_&amp;_Type
- cat_vmm1: Vehicle_Make_and_Model1
- cat_vmm2: Vehicle_Make_and_Model2
- cat_vmy: group 5 years into one label for vehicles manufactured on or before 2010
- cat_vqpt: combination category of qpt and fpt
- cat_vregion: Imported_or_Domestic_Car
- cat_zip: aassured_zip
- int_acc_lia: lia_class
- int_claim_plc: number of claims received by the policy
- int_others: Multiple_Products_with_TmNewa_(Yes_or_No?)
- real_acc_dmg: pdmg_acc
- real_acc_lia: plia_acc
- real_loss_plc: sum of claim paid loss amount by the policy
- real_prem_dmg: policy premium on all damage coverage type
- real_prem_ins: sum of premium by insured's id
- real_prem_lia: policy premium on all liability coverage type
- real_prem_plc: sum of premium by policy number
- real_prem_thf: policy premium on all theft coverage type
- real_prem_vc: sum of premium by vehicle coding
- real_vcost: Replacement_cost_of_insured_vehicle
- real_ved: Engine_Displacement_(Cubic_Centimeter)

# Feature Engineering

Feature engineering will do feature generation and feature selection on Baseline Dataset. Details are as following

## Feature generation

Feature generation involve rebucketing, feature interaction, and mean encoding:

# Model Specification and Assumption

1

#
Contract level information includes 'Main_Insurance_Coverage_Group', 'Insurance_Coverage', 'Insured_Amount1', 'Insured_Amount2', 'Insured_Amount3', 'Coverage_Deductible_if_applied', 'Premium'.

2

#
Claim level information includes 'Main_Insurance_Coverage_Group', 'Paid_Loss_Amount', 'paid_Expenses_Amount', 'Salvage_or_Subrogation?', 'At_Fault?', 'Deductible', 'number_of_claimants', 'Claim_Number'

## Last updated: 08/2018
