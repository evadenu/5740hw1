package nlp.assignments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import nlp.assignments.MaximumEntropyClassifier.EncodedDatum;
import nlp.assignments.MaximumEntropyClassifier.Encoding;
import nlp.assignments.MaximumEntropyClassifier.Factory;
import nlp.assignments.MaximumEntropyClassifier.IndexLinearizer;
import nlp.classify.BasicFeatureVector;
import nlp.classify.BasicLabeledFeatureVector;
import nlp.classify.FeatureExtractor;
import nlp.classify.FeatureVector;
import nlp.classify.LabeledFeatureVector;
import nlp.classify.LabeledInstance;
import nlp.classify.ProbabilisticClassifier;
import nlp.math.DoubleArrays;
import nlp.util.Counter;
import nlp.util.Indexer;

public class PerceptronClassifier<I, F, L> {

	double[] w;
	int iterations;
	FeatureExtractor<I, F> featureExtractor;
	Encoding<F, L> encoding;
	IndexLinearizer indexLinearizer;
	
	public PerceptronClassifier(FeatureExtractor<I, F> fe) {
		featureExtractor = fe;
	}
	
	public void trainPerceptron(List<LabeledInstance<I, L>> trainingData,
			List<LabeledInstance<String, String>> testData){
		
		// build classifier
		// build data encodings so the inner loops can be efficient
		encoding = buildEncoding(trainingData);
		indexLinearizer = buildIndexLinearizer(encoding);
		w = DoubleArrays.constantArray(0.0, indexLinearizer.getNumLinearIndexes());
		EncodedDatum[] data = encodeData(trainingData, encoding);
		

		//Loop through w and adjust if incorrect
		//y*=argmax_y_wy * O(x)
		for(int i = 0; i < data.length; i++){
			int pred = predict(data[i], indexLinearizer);
			int trueLabel = data[i].getLabelIndex();
			if(pred != trueLabel) {
//				System.out.println(pred + " " + trueLabel);
				if(pred != -1) {
					updateWeights(data[i], w, pred, indexLinearizer, -1);
				}
				updateWeights(data[i], w, trueLabel, indexLinearizer, 1);
			} else {
//				System.out.println("correct " + pred + " " + trueLabel);
			}
		}
	}
	
	private void updateWeights(EncodedDatum datum, double[] w, int label, IndexLinearizer indexLinearizer, int sign) {
		for(int n = 0; n < datum.getNumActiveFeatures(); n++) {
			int linIndex = indexLinearizer.getLinearIndex(datum.getFeatureIndex(n), label);
			w[linIndex] += datum.getFeatureCount(n) * sign;
		}
	}
	
	/*
	 * Argmax y
	 */
	private int predict(EncodedDatum datum, IndexLinearizer indexLinearizer) {
		int max_label = -1;
		double max_score = 0;
		
		for(int i = 0; i < indexLinearizer.numLabels; i++) {
			double score = 0;
			
			for(int n = 0; n < datum.getNumActiveFeatures(); n++) {
				int linIndex = indexLinearizer.getLinearIndex(datum.getFeatureIndex(n), i);
				score += w[linIndex] * datum.getFeatureCount(n);
			}
			
			if(score > max_score) {
				max_score = score;
				max_label = i;
			}
		}
		return max_label;
	}
	
	public L getLabel(I name) {
		FeatureVector<F> featureVector = new BasicFeatureVector<F>(
				featureExtractor.extractFeatures(name));
		EncodedDatum encodedDatum = EncodedDatum.encodeDatum(featureVector,
				encoding);
		
		int pred = predict(encodedDatum, indexLinearizer);
		return encoding.getLabel(pred);
	}
	
	private double[] buildInitialWeights(IndexLinearizer indexLinearizer) {
		return DoubleArrays.constantArray(0.0,
				indexLinearizer.getNumLinearIndexes());
	}

	private IndexLinearizer buildIndexLinearizer(Encoding<F, L> encoding) {
		return new IndexLinearizer(encoding.getNumFeatures(),
				encoding.getNumLabels());
	}

	public Encoding<F, L> buildEncoding(List<LabeledInstance<I, L>> data) {
		Indexer<F> featureIndexer = new Indexer<F>();
		Indexer<L> labelIndexer = new Indexer<L>();
		for (LabeledInstance<I, L> labeledInstance : data) {
			L label = labeledInstance.getLabel();
			Counter<F> features = featureExtractor
					.extractFeatures(labeledInstance.getInput());
			LabeledFeatureVector<F, L> labeledDatum = new BasicLabeledFeatureVector<F, L>(
					label, features);
			labelIndexer.add(labeledDatum.getLabel());
			for (F feature : labeledDatum.getFeatures().keySet()) {
				featureIndexer.add(feature);
			}
		}
		return new Encoding<F, L>(featureIndexer, labelIndexer);
	}

	private EncodedDatum[] encodeData(List<LabeledInstance<I, L>> data,
			Encoding<F, L> encoding) {
		EncodedDatum[] encodedData = new EncodedDatum[data.size()];
		for (int i = 0; i < data.size(); i++) {
			LabeledInstance<I, L> labeledInstance = data.get(i);
			L label = labeledInstance.getLabel();
			Counter<F> features = featureExtractor
					.extractFeatures(labeledInstance.getInput());
			LabeledFeatureVector<F, L> labeledFeatureVector = new BasicLabeledFeatureVector<F, L>(
					label, features);
			encodedData[i] = EncodedDatum.encodeLabeledDatum(
					labeledFeatureVector, encoding);
		}
		return encodedData;
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
//		int[] instance = {0,10,20,30,40};
//		PerceptronClassifier<I, F, L>.
//		int[] adjW = buildPerceptron(trainingData,testData);
//		
//		for(int i = 0; i<adjW.length; i++){
//			if(i>0){
//				//System.out.print(",");
//			}
			//System.out.print(adjW[i]);
			
			
//		}
	}

}
