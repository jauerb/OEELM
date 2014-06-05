#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    oeelm.py
    Joshua E. Auerbach (joshua.auerbach@epfl.ch)
    
    Online Extreme Evolutionary Learning Machines
    Copyright © 2014 Joshua E. Auerbach
    
    Laboratory of Intelligent Systems, EPFL
    
    This is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License (GPL)
    as published by the Free Software Foundation, either version 3 of the 
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
"""


import numpy as np
from numpy.random import RandomState 
import sys

iteration = 0

# implementing like this because rng.choice was 
# real slow with replacement = False
def choice_without_replacement(rng, n, size) :
    result = set()
    while len(result) < size :
        result.add(rng.randint(0, n))
    return result


class BinaryFeature(object):
    def __init__(self, weights=None, beta=None):
        self.weights = weights
        self.beta = beta
        self.threshold = len(weights) * beta - list(weights).count(-1)
    
    def dist(self, other):
        return sum(np.asarray(self.weights != other.weights, dtype=int))
    
    @classmethod
    def create_random(cls, rng, num_inputs, beta):
        weights = rng.choice([-1,1], num_inputs)
        return cls(weights, beta)

class Learner(object):
    def __init__(self, features, weights=None):
        self.features = features
        self.feature_weights = np.asarray(np.column_stack((feature.weights.T 
                                          for feature in features)),dtype=float)
        self.feature_thresholds = np.asarray([feature.threshold for 
                                         feature in self.features],dtype=float)
        if weights is None :
            self.weights = np.zeros(len(features) + 1) #include bias
        else:
            self.weights = weights
        
        self.weight_magnitudes_exponential_moving_average = abs(self.weights)
        self.sum_squared_norms = 0.0
        self.squared_norm_ema = 0.0
        
        self.archive_mask = np.ones(len(features) + 1, dtype=bool)
        #bias is always archived (false in mask means archived)
        self.archive_mask[-1] = False 
    
    def get_feature_values(self, input_):
        lin_outputs = np.dot(input_, self.feature_weights)        
        return np.hstack( (np.asarray(np.greater(lin_outputs, 
                                                 self.feature_thresholds), 
                                      dtype=float),
                          np.asarray([1.]) ) )

    def get_sigmoid_feature_values(self, input_):
        lin_outputs = (np.dot(input_, self.feature_weights) - 
                       self.feature_thresholds)        
        return np.hstack( (1. / (1 + np.exp(-1.0 * lin_outputs)),
                          np.asarray([1.]) ) )


    def replace_worst_features(self, rng):
        # select feature with smallest moving average weight, 
        # don't consider the bias
        
        if (params["REPLACEMENT_RATE"] * len(self.features)) < 1 :
            if rng.rand() <= (params["REPLACEMENT_RATE"] * len(self.features)) :
                num_to_replace = 1
            else :
                num_to_replace = 0
        else :
            num_to_replace = np.ceil(
                            params["REPLACEMENT_RATE"] * len(self.features))
        
        fitnesses = self.weight_magnitudes_exponential_moving_average[:-1]
        non_archived_fitnesses = (
          self.weight_magnitudes_exponential_moving_average[self.archive_mask])
        
        median_weight_mag = np.median(non_archived_fitnesses)
        if len(non_archived_fitnesses) == 0 :
            return
        cut_off = np.percentile(non_archived_fitnesses,
                                params["REPLACEMENT_RATE"] * 100)
                                
        if (params["USE_ARCHIVE"] and 
            rng.rand() < (params["NUM_FEATURES"] / 
                          float(params["NUM_ITERATIONS"]))):
            # get best non-archived
            best = np.amax(non_archived_fitnesses)
            # get actual index of this
            best_index = np.where(fitnesses == best)[0][0]            
            self.archive_mask[best_index] = False # False in mask means archived
        
        num_replaced=0
        self.replaced = []
        for index, ema in enumerate(fitnesses) :
            # if "fitness" of the feature is less than the cut off
            # and not archived then will be replaced
            if ema <= cut_off and self.archive_mask[index] : 
                num_replaced += 1
                self.replaced.append(index)
                
                if params["USE_REPRODUCTION"] :
                    # select parents through tournament
                    parent1_index = max(choice_without_replacement(rng, 
                                                params["NUM_FEATURES"], 
                                                size=params["TOURNAMENT_SIZE"]),
                                  key = lambda x:  fitnesses[x])          
                    parent1 = self.features[parent1_index]

                    if rng.rand() <= params["CROSSOVER_PROB"] :
                        parent2_index = parent1_index
                        while (parent2_index == parent1_index) :
                            parent2_index = max(choice_without_replacement(rng, 
                                                params["NUM_FEATURES"], 
                                                size=params["TOURNAMENT_SIZE"]),
                                  key = lambda x:  fitnesses[x])
                        parent2 = self.features[parent2_index]
                        
                        crossover_point = rng.choice(
                                            range(params["NUM_INPUT_VARS"]+1))
                    
                        child_weights = np.concatenate(
                                            (parent1.weights[:crossover_point],
                                             parent2.weights[crossover_point:])
                                                       )
                        child_beta = rng.choice([parent1.beta, parent2.beta])
                    else :
                        child_weights = parent1.weights.copy()
                        child_beta = parent1.beta

                    if params["WEIGHT_MUTATION_PROB"] > 0 :
                        # get 1.0 if > mut_prob else -1.0 
                        # (apply mutation if rand <= mut_prob)
                        mutations = 2.0 * (np.asarray(
                                                rng.rand(len(child_weights)) > 
                                                params["WEIGHT_MUTATION_PROB"],
                                                dtype=float) - 0.5)
                        # apply chosen bit flip mutations
                        child_weights = child_weights * mutations

                    if params["BETA_MUTATION_PROB"] > 0 :
                        # if not keeping BETA constant, then apply 
                        # Gaussian mutation with std dev = BETA_STD
                        # with prob BETA_MUTATION_PROB

                        if rng.rand() <= params["BETA_MUTATION_PROB"] :
                            child_beta = max(0, min(1, 
                                                child_beta + 
                                rng.normal(0,params["BETA_STD"])))
                     
                    new_feature = BinaryFeature(child_weights, child_beta)
                    
                else :
                    # randomly generate a new feature, as is done in 
                    # the Mahmood and Sutton paper
                    new_feature = BinaryFeature.create_random(rng, 
                                                    params["NUM_INPUT_VARS"],
                                                    params["BETA"])
                # replace chosen feature in the feature map
                self.features[index] = new_feature
                
                #update its weights, and threshold
                self.feature_weights[:,index] = new_feature.weights.T            
                self.feature_thresholds[index] = new_feature.threshold
                
                #set its ema to be that of the median
                self.weight_magnitudes_exponential_moving_average[index] = (
                                                        median_weight_mag)
                
                #set its actual weight to zero
                self.weights[index] = 0.0
                
                # need this for the first few updates when weights are 
                # very similar
                if num_replaced >= num_to_replace :
                    break
            

    def get_output(self, input_):
        return np.dot(self.weights, self.get_feature_values(input_))
    
    def get_mse(self, input_, target):
        return (target - self.get_output(input_))**2
    
    def train(self, input_, target, base_learning_rate, iteration):
         
        feature_values = self.get_feature_values(input_)
        
        squared_feature_norm = np.dot(feature_values, feature_values)

        if iteration == 0 :
            self.squared_norm_ema = squared_feature_norm
        else :
            self.squared_norm_ema = (self.squared_norm_ema * 0.999 + 
                                 squared_feature_norm * 0.001)
        learning_rate = base_learning_rate / self.squared_norm_ema
        
        output = np.inner(self.weights, feature_values)
        
        error = target - output
        self.weights = self.weights + learning_rate * error * feature_values
        
        
        if params["USE_BACKPROP"] :
            output_gradient = error # really, this is negative of grad
            
            #compute gradients, as if was using logistic units
            sigmoid_features_vals = self.get_sigmoid_feature_values(input_)
            
            hidden_gradients = (np.sign(output_gradient * self.weights.T) * 
                                sigmoid_features_vals * 
                                (1. - sigmoid_features_vals))

            self.feature_weights += (params["BACKPROP_LEARNING_RATE"] * 
                                      np.dot(input_[:, None], 
                                            hidden_gradients[None,:])[:,:-1])
                                                    
        
        
        if params["USE_SELECTION"] :
            self.weight_magnitudes_exponential_moving_average = (
                params["MOVING_AVERAGE_ALPHA"] * abs(self.weights) +
                (1 - params["MOVING_AVERAGE_ALPHA"])  * 
                self.weight_magnitudes_exponential_moving_average ) 

        return error**2

rng = RandomState(int(sys.argv[1]))
params_file = open(sys.argv[2], "r")
output_file = open(sys.argv[3], "w")

params = { line.split("=")[0].strip() : eval(line.split("=")[1]) 
           for line in open("params/default_params")   }
for line in params_file :
    params[line.split("=")[0].strip()] = eval(line.split("=")[1])
params_file.close()

target_features = [BinaryFeature.create_random(rng, params["NUM_INPUT_VARS"], 
                                               params["BETA"]) 
                   for _ in range(params["NUM_HIDDEN_TARGETS"])]

#make random weights, but don't use bias (this is what Mahmood paper does)
target_weights = np.concatenate((rng.normal(0,1,params["NUM_HIDDEN_TARGETS"]),
                                 [0])) 

target_functions = [Learner(target_features, target_weights)]

if params["NON_STATIONARY"] :
    features_to_replace = choice_without_replacement(rng, 
                                params["NUM_HIDDEN_TARGETS"],
                                params["PERCENT_TARGET_FEATURES_TO_REPLACE"] * 
                                params["NUM_HIDDEN_TARGETS"])
    
    target_features = [BinaryFeature(feature.weights, 
                                     feature.beta) 
                       for feature in target_features]
    target_weights = target_weights.copy()
    for f in features_to_replace :
        target_features[f] = BinaryFeature.create_random(rng, 
                                           params["NUM_INPUT_VARS"], 
                                           params["BETA"])
    target_weights[f] = rng.normal(0,1)
    
    target_functions.append(Learner(target_features, target_weights))


features =[BinaryFeature.create_random(rng, params["NUM_INPUT_VARS"],
                                       params["BETA"]) 
                   for _ in range(params["NUM_FEATURES"])]

learner = Learner(features)

sum_of_errors = 0.0

mse_history = np.zeros(10000)
ema_history = np.zeros(10000)
exponential_moving_avg = 0.0

for iteration in range(params["NUM_ITERATIONS"]) :

    if params["CONTINUOUS_INPUTS"] :
        random_input = rng.rand(params["NUM_INPUT_VARS"])
    else :
        random_input = rng.choice([0,1],  params["NUM_INPUT_VARS"])

    if params["NON_STATIONARY"] :
        target_val = (target_functions[(iteration / 100000) % 2
                                      ].get_output(random_input) + 
                                            rng.normal(0,1))
    else :
        target_val = (target_functions[0].get_output(random_input) + 
                                            rng.normal(0,1))
    
    mse = learner.train(random_input, target_val, params["BASE_LEARNING_RATE"], 
                        iteration)
    if iteration == 0 :
        exponential_moving_avg = mse
    else :
        exponential_moving_avg = exponential_moving_avg * 0.999 + mse*0.001
    sum_of_errors += mse
    mse_history[iteration%10000] = mse
    ema_history[iteration%10000] = exponential_moving_avg
    
    if ( params["USE_SELECTION"] ) :
        learner.replace_worst_features(rng)

    if (iteration + 1) % 100 == 0 :
        if iteration < 10000:
            rolling_avg_mse = np.mean(mse_history[0:iteration])
            rolling_avg_ema = np.mean(ema_history[0:iteration])
        else :
            rolling_avg_mse = np.mean(mse_history)
            rolling_avg_ema = np.mean(ema_history)
        output_file.write(str(iteration + 1) + "," + 
              str(mse) + "," +
              str(sum_of_errors/(iteration + 1.0)) + "," +
              str(exponential_moving_avg) + "," +
              str(rolling_avg_mse) + "," +
              str(rolling_avg_ema) + "," +
              str(min(learner.weight_magnitudes_exponential_moving_average)) + 
              "," +
              str(np.median(
                    learner.weight_magnitudes_exponential_moving_average)) + 
              "," +
              str(max(learner.weight_magnitudes_exponential_moving_average)) + 
              "\n")
    if (iteration + 1) % 10000 == 0 :
        output_file.flush()
output_file.close()


