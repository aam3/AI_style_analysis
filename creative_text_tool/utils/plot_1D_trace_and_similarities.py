import matplotlib.pyplot as plt
import numpy as np

def plot_1D_trace_and_similarities(x_number_of_sentences, docs_to_analyze, colors, labels, window_size=1):

    # Create figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot 1: Covariance Tract
    for i,doc in enumerate(docs_to_analyze[0:4]):
        
        ax1.plot(x_number_of_sentences, [doc.sentence_stats[j]['cov_trace'] for j in x_number_of_sentences], 
                 color=colors[i], label=labels[i])
    
    ax1.set_title('Covariance Trace')
    ax1.set_xlabel('Number of Sentences')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(False)
    
    
    # Plot 2: Variance of COV matrix diagonal
    for i,doc in enumerate(docs_to_analyze[0:4]):
        
        ax2.plot(x_number_of_sentences, [doc.sentence_stats[j]['cov_diag_var'] for j in x_number_of_sentences], 
                 color=colors[i], label=labels[i])
    
    ax2.set_title('Cov Matrix - Variance')
    ax2.set_xlabel('Number of Sentences')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(False)
    
    # Plot 3: Similarity STD
    window_size = 2
    for i,doc in enumerate(docs_to_analyze[0:4]):
        
        ax3.plot(x_number_of_sentences, [doc.sentence_stats[j][window_size]['mean_sentence_similarity'] for j in x_number_of_sentences], 
                 color=colors[i], label=labels[i])
    
        ax3.fill_between(x_number_of_sentences, np.subtract([doc.sentence_stats[j][window_size]['mean_sentence_similarity'] for j in x_number_of_sentences], [doc.sentence_stats[j][window_size]['std_sentence_similarity'] for j in x_number_of_sentences]), 
                        np.add([doc.sentence_stats[j][window_size]['mean_sentence_similarity'] for j in x_number_of_sentences], [doc.sentence_stats[j][window_size]['std_sentence_similarity'] for j in x_number_of_sentences]), 
                        color=colors[i], alpha=0.2)
        
    
    ax3.set_title('Similarity MEAN')
    ax3.set_xlabel('Number of Sentences')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(False)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Display the plots
    plt.show()