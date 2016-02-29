package nlp.assignments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import nlp.classify.FeatureExtractor;
import nlp.classify.LabeledInstance;
import nlp.classify.ProbabilisticClassifier;
import nlp.util.Counter;

public class PerceptronTrainer {

	public static int[] buildPerceptron(List<LabeledInstance<String[], String>> trainingData,List<LabeledInstance<String[], String>> testData){
		//Initialize Weight Vector with 0
		int[] w = new int[trainingData.size()];
		
		for(LabeledInstance f : trainingData){
			
			System.out.println(f.getLabel());
		}
		
		// build classifier
		FeatureExtractor<String[], String> featureExtractor = new FeatureExtractor<String[], String>() {
			public Counter<String> extractFeatures(String[] featureArray) {
				return new Counter<String>(Arrays.asList(featureArray));
			}
		};
	MaximumEntropyClassifier.Factory<String[], String, String> maximumEntropyClassifierFactory = new MaximumEntropyClassifier.Factory<String[], String, String>(
				1.0, 20, featureExtractor);
		
		ProbabilisticClassifier<String[], String> maximumEntropyClassifier = maximumEntropyClassifierFactory
			.trainClassifier(trainingData);
		System.out.println("Probabilities on test instance: "+ maximumEntropyClassifier.getProbabilities(datum4.getInput()));
		MaximumEntropyClassifier.Factory<String[], String, String> maximumEntropyClassifierFactory = new MaximumEntropyClassifier.Factory<String[], String, String>(1.0, 20, featureExtractor);
		
		
		//Loop through w and adjust if incorrect
		//y*=argmax_y_wy * O(x)
		for(int i = 0; i<w.length;i++){
			MaximumEntropyClassifier.ObjectiveFunction<F, L>
			
		}
		
		return w;
	}
	
	
	
	public static void main(String[] args) {


		// create datums
		LabeledInstance<String[], String> datum1 = new LabeledInstance<String[], String>(
				"cat", new String[] { "fuzzy", "claws", "small" });
		LabeledInstance<String[], String> datum2 = new LabeledInstance<String[], String>(
				"bear", new String[] { "fuzzy", "claws", "big" });
		LabeledInstance<String[], String> datum3 = new LabeledInstance<String[], String>(
				"cat", new String[] { "claws", "medium" });
		
		
		LabeledInstance<String[], String> datum4 = new LabeledInstance<String[], String>(
				"cat", new String[] { "claws", "small" });
			
		// create training set
		List<LabeledInstance<String[], String>> trainingData = new ArrayList<LabeledInstance<String[], String>>();
		trainingData.add(datum1);
		trainingData.add(datum2);
		trainingData.add(datum3);

		// create test set
		List<LabeledInstance<String[], String>> testData = new ArrayList<LabeledInstance<String[], String>>();
		testData.add(datum4);
//
//		// build classifier
//		FeatureExtractor<String[], String> featureExtractor = new FeatureExtractor<String[], String>() {
//			public Counter<String> extractFeatures(String[] featureArray) {
//				return new Counter<String>(Arrays.asList(featureArray));
//			}
//		};
//		MaximumEntropyClassifier.Factory<String[], String, String> maximumEntropyClassifierFactory = new MaximumEntropyClassifier.Factory<String[], String, String>(
//				1.0, 20, featureExtractor);
		
//		ProbabilisticClassifier<String[], String> maximumEntropyClassifier = maximumEntropyClassifierFactory
//				.trainClassifier(trainingData);
//		System.out.println("Probabilities on test instance: "
//				+ maximumEntropyClassifier.getProbabilities(datum4.getInput()));


		//Can use the log probabilities
		
		// Perceptron 
		int[] instance = {0,10,20,30,40};	
		int[] adjW = buildPerceptron(trainingData,testData);
		
		for(int i = 0; i<adjW.length; i++){
//			if(i>0){
//				//System.out.print(",");
//			}
			//System.out.print(adjW[i]);
			
			
		}
	}

}
