# load packages
using Pkg, CSV, DataFrames, Flux, CuArrays, BrowseTables, Statistics, StatsBase, StatsPlots, MLJ, MLJBase, MLBase, ROC, Dates, LinearAlgebra, DecisionTree, StatsPlots.PlotMeasures
using BSON: @save, @load
# -------------------------------- import data ------------------------------- #
df = DataFrame!(CSV.File(raw"C:\kaggle\titanic\data\train.csv"));
select!(df, Not([:Name, :Ticket, :Cabin, :PassengerId]));

# ---------------------- data processing and imputation ---------------------- #
df[!, :Sex] = [if occursin("female", x) 1 else 0 end for x in df.Sex];
const medianage = median(skipmissing(df.Age));
df[!, :missing_age] = [if ismissing(x) 1 else 0 end for x in df.Age];
df[!, :Age] = [if ismissing(x) medianage else x end for x in df.Age];
df[!, :Embarked] = [if ismissing(x) "S" else x end for x in df.Embarked];
df[!, :embarked_s] = [if occursin("S", x) 1 else 0 end for x in df.Embarked];
df[!, :embarked_c] = [if occursin("C", x) 1 else 0 end for x in df.Embarked];
df[!, :embarked_q] = [if occursin("Q", x) 1 else 0 end for x in df.Embarked];
select!(df, Not(:Embarked));

# ---------------------------------------------------------------------------- #
#                           evaluate on training data                          #
# ---------------------------------------------------------------------------- #

# ----------------- split data into three randomized groups ---------------- #
Y = df[:, :Survived];

index = sample(collect(1:1:length(Y)), length(Y); replace=false);
train_index = index[1:534];
val_index = index[535:535+178];
test_index = index[535+178+1:end];

y_train = Y[train_index];
y_val = Y[val_index];
y_test = Y[test_index];

X = Matrix(df[:, 2:end]);
x_train = X[train_index, :];
x_val = X[val_index, :];;
x_test = X[test_index, :];

# ---------------------------- standardize X data ---------------------------- #
xtrain_standardizer = StatsBase.fit(ZScoreTransform, x_train, dims=1);
x_train_std = permutedims(StatsBase.transform!(xtrain_standardizer, x_train)); 
x_val_std = permutedims(StatsBase.transform!(xtrain_standardizer, x_val));
x_test_std = permutedims(StatsBase.transform!(xtrain_standardizer, x_test));

function onehotmatrix(inputvector)
    firstcol = Int64[];
    secondcol = Int64[];
    for c in inputvector
        if c == 0
            push!(firstcol, 1)
            push!(secondcol, 0)
        else
            push!(firstcol, 0)
            push!(secondcol, 1)
        end
    end
    return permutedims(hcat(firstcol, secondcol))
end

# ------------------------ train model on x_train_std ------------------------ #
opt = ADAM(0.0001, (0.9, 0.999));
m1 = Chain(Dense(10, 10, NNlib.mish), Dense(10, 10, NNlib.mish), Dense(10, 2, NNlib.mish), softmax);

y_train_onehot2 = onehotmatrix(y_train);
y_val_onehot2 = onehotmatrix(y_val);
y_test_onehot2 = onehotmatrix(y_test);

training_data = [(x_train_std, y_train_onehot2)];

# training_data2 = Flux.Data.DataLoader(x_train_std, y_train_onehot2, batchsize=32, shuffle=true);

function crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight=1)
    -sum(y .* log.(ŷ) .* weight)
end

loss(x,y) = crossentropy(m1(x), y, weight=[1,1]) + sum(norm, Flux.params(m1)) # weight is a two element array, higher value in first element penalizes false positives, higher value in second element penalizes false negatives
  
#loss(x,y) =  sum(Flux.binarycrossentropy.(m(x),y)) + 0.5*sum(norm, Flux.params(m));
# callback() = @show(loss(x_train_std, y_train));

epoch_vector = Int64[];
training_loss_vector = Float64[];
val_loss_vector = Float64[];
training_roc_auc_vector = Float64[];
val_roc_auc_vector = Float64[];

println("evaluating on training dataset")

function training_on_dataset(xtrain, xval, ytrain, yval)
    for epoch in 1:5000
        Flux.train!(loss, Flux.params(m1), training_data, opt)
        push!(epoch_vector, epoch)
        training_loss = loss(xtrain, ytrain);
        training_roc = ROC.AUC(ROC.roc(m1(xtrain)[2,:], ytrain[2,:]));
        val_loss = loss(xval, yval);
        val_roc = ROC.AUC(ROC.roc(m1(xval)[2,:], yval[2,:]));
        push!(training_loss_vector, training_loss)
        push!(training_roc_auc_vector, training_roc)
        push!(val_loss_vector, val_loss)
        push!(val_roc_auc_vector, val_roc)
        if epoch % 100 == 0
            print(" $epoch ")
        end
    end
end

training_on_dataset(x_train_std, x_val_std, y_train_onehot2, y_val_onehot2)

lossplot = plot(epoch_vector, training_loss_vector, label="Training loss", lw=2, color=:blue, legend=(0.8, 0.3), title="Loss", ylabel="Training loss", xlabel="No. of epochs");
plot!(twinx(), epoch_vector, val_loss_vector, label="Validation loss",color=:orange, lw=2, ylabel="Validation loss", legend=(0.8, 0.2))

rocplot = plot(epoch_vector, training_roc_auc_vector, label="Training ROC AUC", lw=2, legend=:bottomright, title="ROC AUC", xlabel="No. of epochs", ylabel="ROCAUC");
plot!(epoch_vector, val_roc_auc_vector, label="Validation ROC AUC", lw=2);

println("training complete -- ", Dates.now())

combinedplot = plot(lossplot, rocplot, layout=(1,2), size=(1500,1000), margin=15mm)

display(combinedplot)
# savefig(combinedplot, "titanic_flux_adam0001_5000epochs_onehiddenlayer_regularized_08JUL2020.png")
# @save "titanic_flux_adam0001_5000epochs_onehiddenlayer_regularized_08JUL2020.bson" m

ROC.AUC(ROC.roc(m1(x_test_std)[2,:], y_test_onehot2[2,:]))

# ---------------------------------------------------------------------------- #
#                      fit model to whole training dataset                     #
# ---------------------------------------------------------------------------- #
# @load raw"C:\Julia\kaggle_titanic\titanic_flux_adam0001_5000epochs_onehiddenlayer_regularized_08JUL2020.bson" m;
println("starting training on whole dataset")
m2 = Chain(Dense(10, 10, NNlib.mish), Dense(10, 10, NNlib.mish), Dense(10, 2, NNlib.mish), softmax);

xall_standardizer = StatsBase.fit(ZScoreTransform, X, dims=1);
x_all = permutedims(StatsBase.transform!(xall_standardizer, X)); 
y_all_onehot2 = onehotmatrix(Y);

all_training = [(x_all, y_all_onehot2)];

function crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight=1)
    -sum(y .* log.(ŷ) .* weight)
end

loss(x,y) = crossentropy(m2(x), y, weight=[1,1]) + sum(norm, Flux.params(m2));
opt = ADAM(0.0001, (0.9, 0.999));

for epoch in 1:3500
    Flux.train!(loss, Flux.params(m2), all_training, opt)
    if epoch % 100 == 0
        print(" $epoch ")
    end
end

@save "titanic_flux_alltraining_adam0001_35000epochs_onehiddenlayer_regularized_10JUL2020.bson" m2

# ----------------------- make predictions on test.csv ----------------------- #
dftest = DataFrame!(CSV.File(raw"C:\kaggle\titanic\data\test.csv"));
testpassengerids = dftest.PassengerId;
select!(dftest, Not([:Name, :Ticket, :Cabin, :PassengerId]));

dftest[!, :Sex] = [if occursin("female", x) 1 else 0 end for x in dftest.Sex];
const medianage2 = median(skipmissing(dftest.Age));
const medianfare = median(skipmissing(dftest.Fare));
dftest[!, :missing_age] = [if ismissing(x) 1 else 0 end for x in dftest.Age];
dftest[!, :Age] = [if ismissing(x) medianage2 else x end for x in dftest.Age];
dftest[!, :Fare] = [if ismissing(x) medianfare else x end for x in dftest.Fare];
dftest[!, :Embarked] = [if ismissing(x) "S" else x end for x in dftest.Embarked];
dftest[!, :embarked_s] = [if occursin("S", x) 1 else 0 end for x in dftest.Embarked];
dftest[!, :embarked_c] = [if occursin("C", x) 1 else 0 end for x in dftest.Embarked];
dftest[!, :embarked_q] = [if occursin("Q", x) 1 else 0 end for x in dftest.Embarked];
select!(dftest, Not(:Embarked));

xfinaltest = Matrix(dftest);
xfintaltest_std = permutedims(StatsBase.transform!(xall_standardizer, xfinaltest));

finalpred = [ifelse(x >= 0.5, 1, 0) for x in m2(xfintaltest_std)[2,:]];


dfinalprediction = DataFrame(PassengerId = testpassengerids, Survived = finalpred);

# write output to csv
CSV.write("titanic_flux_adam0001_3500epochs_onehiddenlayer_regularized_10JUL2020.csv", dfinalprediction)

# ---------------------------------------------------------------------------- #


#                    attempt random forest classification                     #
# ---------------------------------------------------------------------------- #

# n_subfeatures=-1; n_trees=50; partial_sampling=0.7; max_depth=-1;
# min_samples_leaf=5; min_samples_split=2; min_purity_increase=0.0;

# model    =   build_forest(y_train, permutedims(x_train_std),
#                           n_subfeatures,
#                           n_trees,
#                           partial_sampling,
#                           max_depth,
#                           min_samples_leaf,
#                           min_samples_split,
#                           min_purity_increase);

# forestprediction = apply_forest(model, permutedims(x_val_std));

# forestpredictionproba = apply_forest_proba(model, permutedims(x_val_std), [0,1])[:, 2];
# forestpredictionproba_training = apply_forest_proba(model, permutedims(x_train_std), [0,1])[:, 2];
# forestpredictionproba_testing = apply_forest_proba(model, permutedims(x_test_std), [0,1])[:, 2];


# println("training ROC -- ", ROC.AUC(ROC.roc(forestpredictionproba_training, y_train)))
# println("validation ROC -- ", ROC.AUC(ROC.roc(forestpredictionproba, y_val)))
# println("testing ROC -- ", ROC.AUC(ROC.roc(forestpredictionproba_testing, y_test)))