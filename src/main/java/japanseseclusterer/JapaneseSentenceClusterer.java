package japanseseclusterer;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.vectorizer.SparseVectorsFromSequenceFiles;
import org.apache.mahout.math.NamedVector;
import org.atilika.kuromoji.Token;
import org.atilika.kuromoji.Tokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Clusters Japanese sentences according to topics 
 */
public class JapaneseSentenceClusterer {
	
	private static final Logger log = LoggerFactory.getLogger(SparseVectorsFromSequenceFiles.class);
	
	private Configuration conf = new Configuration();
	
	private FileSystem fs;
	
	private Path inputFile;
	
	private Path outputDir;
	
	/** 
	 * Constructor - initializes FileSystem and clears the output folder
	 * 
	 * @param inputFile Path to file with one sentence per line
	 * @param outputDir Path to output directory
	 */
	public JapaneseSentenceClusterer(Path inputFile, Path outputDir) {
		
		this.inputFile = inputFile;
		this.outputDir = outputDir;		
		
		try {
			this.fs = FileSystem.get(conf);
			// delete old files
			fs.delete(outputDir, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	/**
	 * Starts all steps from input files to clustering
	 */
	public void run() {
		List<String> sentencesText = readFile();
		processText(sentencesText);   // Tokenize, filter nouns, sparse vectors (tf-id ...)
		cluster();
	}
	
	/**
	 * Reads the input file and stores the lines in an array
	 */
	private List<String> readFile() {
				
		String line;
				
		BufferedReader br;
		
		List<String> sentences = new ArrayList<String>();
		
		try {
			br = new BufferedReader(new FileReader(inputFile.toString()));
						
			int lineNumber = 1;
			while ((line = br.readLine()) != null) {
				
				sentences.add(line);
				lineNumber++;
				if(lineNumber > 500) {	// TODO: remove after testing
					break;
				}
			}
			br.close();
	
		}catch (FileNotFoundException e) {
			log.error("Could not find input file");
		}
		catch(IOException e) {
			log.error("IOException", e);
		}
		
		return sentences;		
		
	}
	
	/**
	 * Tokenizes the sentences with Kuromoji and saves them as sequence files.
	 * Then generates vectors (tf, tfidf), dictionary etc.
	 */
	private void processText(List<String> sentences) {
		
		Tokenizer tokenizer = Tokenizer.builder().build();
		
		SequenceFile.Writer writer;
		
		try {
		
			writer = new SequenceFile.Writer(fs, conf, new Path(outputDir.toString()+"/seqfiles"), Text.class, Text.class);
					
			StringBuilder nouns = new StringBuilder();
			
			int sentenceNum = 0;
			
			for (String rawSentence : sentences) {
				
				for (Token token : tokenizer.tokenize(rawSentence)) {
			    	String posTags = token.getPartOfSpeech();
			    	
			    	if(posTags.startsWith("名詞")) {  // is there a better way to filter for nouns?
			    		nouns.append(token.getBaseForm()+" ");
			    	}
			    }
				writer.append(new Text("sentence"+sentenceNum++), new Text(nouns.toString()));
				//writer.append(new Text("Satz"+sentenceNum++), new Text(rawSentence));    // english
			}
			writer.close();	
			
			SparseVectorsFromSequenceFiles svf = new SparseVectorsFromSequenceFiles();
			
			// -seq -> SequentialAccessSparseVector (best for k-means) | maxDFPercent -> can be used to filter stop words
			svf.run(new String[]{"-i", outputDir.toString()+"/seqfiles", "-o", outputDir.toString()+"/vectors", "-seq", "--maxDFPercent", "85", "--namedVector"});
			
			// TODO: add normalization		
			
		} catch(IOException e) {
			log.error("IOException", e);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Does the actual clustering using k-means with an initial random seed
	 */
	private void cluster() {
		
		try {
			RandomSeedGenerator.buildRandom(conf, new Path(outputDir.toString()+"/vectors/tfidf-vectors"), new Path(outputDir.toString()+"/cluster"), 20, new EuclideanDistanceMeasure());
		
			// params: conf, input path, cluster path, output path, convergenceDelta, maxIterations, run clustering, classification threshold, runsequential
			KMeansDriver.run(conf, new Path(outputDir.toString()+"/vectors/tfidf-vectors"), new Path(outputDir.toString()+"/cluster"), new Path(outputDir.toString()+"/result"), 0.001, 10, true, 0.0, false);
			
		} catch (IOException e) {
			log.error("IOException", e);
		} catch (Exception e) {
			e.printStackTrace();
		}
	
	}
	
	/**
	 * Prints the assignment of sentences to clusters (work in progress)
	 * (Alternatively use Mahout's clusterdumper tool, which shows the top terms per cluster etc.)
	 * 
	 */
	private void printResults() {
		SequenceFile.Reader reader;
		try {
			reader = new SequenceFile.Reader(fs, new Path(outputDir.toString()+"/result/clusteredPoints/part-m-00000"), conf);
		
			
			IntWritable key = new IntWritable();
			WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();
			
			// cluster number, sentences id
			HashMap<Integer, List<String>> cluster = new HashMap<Integer, List<String>>();
			
			while (reader.next(key, value)) {
				NamedVector vec = (NamedVector) value.getVector();
				
				// create entry for cluster, if it does not exist yet
				if(!cluster.containsKey(key.get())) { 
					cluster.put(key.get(), new ArrayList<String>());
				}
				
				cluster.get(key.get()).add(vec.getName());

			}
			reader.close();
			
			Iterator iterator = cluster.keySet().iterator();
			while(iterator.hasNext()) {
				Integer clusterId = (Integer)iterator.next();
				System.out.println("Cluster "+clusterId);
				List<String> sentences = (List<String>) cluster.get(clusterId);
				
				for(String sentence : sentences) {
					System.out.print(sentence+" ");  // TODO: print the sentence itself, instead of just the id
				}
				System.out.println("\n");
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public static void main(String[] args) {
				
		JapaneseSentenceClusterer clusterer = new JapaneseSentenceClusterer(new Path("data/input/examples.utf"), new Path("data/output"));
		
		clusterer.run();
		
		clusterer.printResults();
		
	}

}