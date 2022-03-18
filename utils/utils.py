import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import urllib


offers_training_df = pd.read_parquet('data/offers_training.parquet')
offers_test_df = pd.read_parquet('data/offers_test.parquet')

def plot_images(product):
    
    # Data
    images = product['image_urls']
    
    # Plot it!
    fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(12, 4), dpi=100)
    
    if len(images) > 1:     
        axes = axes.flatten()
        for i, axis in enumerate(axes):
            url = images[i]
            image = np.array(Image.open(urllib.request.urlopen(url)))
            axis.imshow(image)
            axis.axis('off')
    else:
        url = images[0]
        image = np.array(Image.open(urllib.request.urlopen(url)))
        axes.imshow(image)
        axes.axis('off')

    fig.tight_layout()
    plt.show()

def explore_match(match, groundtruth=None):
    """ Explore a match with the offers' images """
    
    # get offer ids
    zal_offer_id = match['zalando']
    comp_offer_id = match['aboutyou']

    if groundtruth is not None:
        true_offer_id = groundtruth.loc[
	        groundtruth['zalando'] == zal_offer_id
            ]['aboutyou'].values[0]
        true_offer = offers_training_df[offers_training_df['offer_id'] == true_offer_id].iloc[0]
    
    # get offers
    zalando_offer = offers_training_df[offers_training_df['offer_id'] == zal_offer_id].iloc[0]
    comp_offer = offers_training_df[offers_training_df['offer_id'] == comp_offer_id].iloc[0]
    
    # show images and text
    print(f"Zalando: BRAND={zalando_offer['brand']}\
                     TITLE={zalando_offer['title']}\
                     COLOR={zalando_offer['color']}\
                     PRICE={zalando_offer['price']}")
    plot_images(zalando_offer)
    
    print(f"Predicted Aboutyou: BRAND={comp_offer['brand']}\
                     TITLE={comp_offer['title']}\
                     COLOR={comp_offer['color']}\
                     PRICE={comp_offer['price']}")    
    plot_images(comp_offer)

    if groundtruth is None:
        return
    elif groundtruth.loc[
	groundtruth['zalando'] == zal_offer_id
    ]['aboutyou'].values[0] == comp_offer_id:
        print('Prediction correct')
    else:
        print('Prediction non correct, actual prediction:')
        print(f"True Aboutyou: BRAND={true_offer['brand']}\
                     TITLE={true_offer['title']}\
                     COLOR={true_offer['color']}\
                     PRICE={true_offer['price']}")  
        plot_images(true_offer)

def presence_in_n(best_n_matches, true_matches):
    all_zal_ids = set(best_n_matches['zalando'].values)
    df_predicted_matches = pd.DataFrame(columns=['zalando', 'success', 'p3', 'p5', 'p10'])
    for id in all_zal_ids:
        groundtruth_match = true_matches.loc[
            true_matches['zalando'] == id
        ]['aboutyou'].values[0]
        
        predicted_n_matches = list(best_n_matches.loc[
            best_n_matches['zalando'] == id
        ]['aboutyou'].values)
        
        p_success = groundtruth_match == predicted_n_matches[0]
        p_in_3 = groundtruth_match in predicted_n_matches[:3]
        p_in_5 = groundtruth_match in predicted_n_matches[:5]
        p_in_10 = groundtruth_match in predicted_n_matches[:10]
        df_predicted_matches = df_predicted_matches.append(
            {'zalando': id, 'success': p_success, 'p3': p_in_3, 'p5': p_in_5, 'p10': p_in_10}, ignore_index=True
        )
    return df_predicted_matches

def get_metrics(true_matches, predicted_matches, best_n_matches, offers_comp, all_true_matches):
    """ Calculate performance metrics """
    
    # True Positives
    TP = len(
        true_matches.merge(
            predicted_matches, 
            on=['zalando', 'aboutyou'], 
            how='inner', 
        )
    )
    
    # False Negatives
    FN = len(true_matches) - TP
    
    # Actual Positives
    positives = len(true_matches)
    assert positives == TP + FN
    
    # Actual Negatives (with respect to the competitor)
    negatives = len(offers_comp) - positives
    
    # Actual negative values (with respect to the competitor)
    offers_comp_with_matches = offers_comp.merge(
        true_matches, 
        left_on='offer_id',
        right_on='aboutyou',
        how='outer',
        indicator=True
    )
    negative_values = offers_comp_with_matches[
        offers_comp_with_matches['_merge'] == 'left_only'
    ]['offer_id'].unique()
    
    assert negatives == len(negative_values)
    
    # Competitor predictions
    comp_preds = predicted_matches['aboutyou'].unique()
    
    # False Negatives (with respect to the competitor)
    FP = len(np.intersect1d(negative_values, comp_preds))
    
    # True Negatives
    TN = negatives - FP
    
    # Precision, Recall and F1 metrics
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 0
    if precision + recall > 0:
        F1 = 2 * precision * recall / (precision + recall)
    
    predicted = presence_in_n(best_n_matches, all_true_matches)
    l = len(predicted)
    l1 = len(predicted.loc[predicted['success']]) * 100
    l3 = len(predicted.loc[predicted['p3']]) * 100
    l5 = len(predicted.loc[predicted['p5']]) * 100
    l10 = len(predicted.loc[predicted['p10']]) * 100
    
    metrics = dict(
        TP=TP,
        FN=FN,
        FP=FP,
        TN=TN,
        positives=positives,
        negatives=negatives,
        precision=precision,
        recall=recall,
        F1=F1,
        PS=round(l1 / l, 4),
        P3=round(l3 / l, 4),
        P5=round(l5 / l, 4),
        P10=round(l10 / l, 4)
    )
        
    return metrics

