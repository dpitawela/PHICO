import numpy as np
import ast

@np.errstate(all='warn')
def label_performance(prediction, user, model, target):
    results = {}
    for class_label in np.unique(ar=target):
        class_mask = target == class_label

        t_class = target[class_mask]
        u_class = user[class_mask]
        m_class = model[class_mask]
        p_class = prediction[class_mask]

        # measures for user
        user_correct_mask = u_class == t_class
        user_wrong_mask = u_class != t_class

        n_user_correct = np.sum(user_correct_mask)
        n_user_wrong = np.sum(user_wrong_mask)
        
        user_final_correct = np.sum(p_class[user_wrong_mask] == t_class[user_wrong_mask])
        user_final_wrong = np.sum(p_class[user_correct_mask] != t_class[user_correct_mask])

        # measures for the model
        model_correct_mask = m_class == t_class
        model_wrong_mask = m_class != t_class

        n_model_correct = np.sum(model_correct_mask)
        n_model_wrong = np.sum(model_wrong_mask)

        model_final_correct = np.sum(p_class[model_wrong_mask] == t_class[model_wrong_mask])
        model_final_wrong = np.sum(p_class[model_correct_mask] != t_class[model_correct_mask])

        results[class_label] = {'u_error': n_user_wrong, 'u_corrected': user_final_correct, 'u_label_correction': user_final_correct/n_user_wrong,
                                'u_correct': n_user_correct, 'u_correction_error': user_final_wrong, 'u_label_error': user_final_wrong/n_user_correct,

                                'm_error': n_model_wrong, 'm_corrected': model_final_correct, 'm_label_correction': model_final_correct/n_model_wrong,
                                'm_correct': n_model_correct, 'm_correction_error': model_final_wrong, 'm_label_error': model_final_wrong/n_model_correct}
    return results

@np.errstate(all='warn')
def overall_performance(prediction, user, model, target):
    label_result=label_performance(prediction, user, model, target)
    u_n_correct_interventions = 0
    u_n_error_interventions = 0
    m_n_correct_interventions = 0
    m_n_error_interventions = 0
    result = {}
    for class_label in label_result.keys():
        u_n_correct_interventions += label_result[class_label]['u_corrected']
        u_n_error_interventions += label_result[class_label]['u_correction_error']

        m_n_correct_interventions += label_result[class_label]['m_corrected']
        m_n_error_interventions += label_result[class_label]['m_correction_error']

    n_interventions_for_u = u_n_correct_interventions + u_n_error_interventions
    n_interventions_for_m = m_n_correct_interventions + m_n_error_interventions
    n_interventions = n_interventions_for_u + n_interventions_for_m

    result['cr_for_u'] = u_n_correct_interventions/n_interventions_for_u
    result['er_for_u'] = u_n_error_interventions/n_interventions_for_u
    result['n_correct_interventions_u'] = u_n_correct_interventions
    result['n_error_interventions_u'] = u_n_error_interventions
    result['n_interventions_u'] = n_interventions_for_u

    result['cr_for_m'] = m_n_correct_interventions/n_interventions_for_m
    result['er_for_m'] = m_n_error_interventions/n_interventions_for_m
    result['n_correct_interventions_m'] = m_n_correct_interventions
    result['n_error_interventions_m'] = m_n_error_interventions
    result['n_interventions_m'] = n_interventions_for_m

    result['cr'] = (m_n_correct_interventions+u_n_correct_interventions)/n_interventions
    result['er'] = (m_n_error_interventions+u_n_error_interventions)/n_interventions
    result['n_correct_interventions'] = m_n_correct_interventions + u_n_correct_interventions
    result['n_error_interventions'] = m_n_error_interventions + u_n_error_interventions
    result['n_interventions'] = n_interventions

    return result, label_result

@np.errstate(all='warn')
def alterationsModel(label_result):
    """This method calculates positive and negative alterations occured on every class for all users
    returns: positive alterations, negative alterations"""
    u_error = 0
    u_corrected = 0
    u_correct = 0
    u_correction_error = 0

    for class_label in label_result.keys():
        try:
            u_error += sum(label_result[class_label].apply(lambda col: col.get('u_error')))
            u_corrected += sum(label_result[class_label].apply(lambda col: col.get('u_corrected')))

            u_correct += sum(label_result[class_label].apply(lambda col: col.get('u_correct')))
            u_correction_error += sum(label_result[class_label].apply(lambda col: col.get('u_correction_error')))

        except: # if data coming from a saved csv
            u_error += sum(label_result[class_label].apply(lambda col: ast.literal_eval(col.replace('nan', '0')).get('u_error')))
            u_corrected += sum(label_result[class_label].apply(lambda col: ast.literal_eval(col.replace('nan', '0')).get('u_corrected')))

            u_correct += sum(label_result[class_label].apply(lambda col: ast.literal_eval(col.replace('nan', '0')).get('u_correct')))
            u_correction_error += sum(label_result[class_label].apply(lambda col: ast.literal_eval(col.replace('nan', '0')).get('u_correction_error')))

    pos = u_corrected/u_error
    neg = u_correction_error/u_correct

    return pos, neg

def meanAlterationsAnnotators(label_result):
    """This method calculates positive and negative alterations occured on every class for all users and returns mean
    returns: positive alterations, negative alterations"""
    u_error = 0
    u_corrected = 0
    u_correct = 0
    u_correction_error = 0

    try:
        u_error = label_result.apply(lambda row: sum([d['u_error'] for d in row]), axis=1)
        u_corrected = label_result.apply(lambda row: sum([d['u_corrected'] for d in row]), axis=1)
        u_correct = label_result.apply(lambda row: sum([d['u_correct'] for d in row]), axis=1)
        u_correction_error = label_result.apply(lambda row: sum([d['u_correction_error'] for d in row]), axis=1)

    except: # if data coming from a saved csv
        u_error = label_result.apply(lambda row: sum([ast.literal_eval(d.replace('nan', '0')).get('u_error') for d in row]), axis=1)
        u_corrected = label_result.apply(lambda row: sum([ast.literal_eval(d.replace('nan', '0')).get('u_corrected') for d in row]), axis=1)
        u_correct = label_result.apply(lambda row: sum([ast.literal_eval(d.replace('nan', '0')).get('u_correct') for d in row]), axis=1)
        u_correction_error = label_result.apply(lambda row: sum([ast.literal_eval(d.replace('nan', '0')).get('u_correction_error') for d in row]), axis=1)
    
    return np.mean(u_corrected/u_error), np.mean(u_correction_error/u_correct)

def modelChoices(raw_outputs):
    """This method calculates choices of model given base, human being correct(c)/not correct(nc)
    returns: a dictionary with results"""
    results = {
        'b:c, h:nc, t:c': sum(raw_outputs.apply(lambda row: sum([1 for h, b, pred, gt in zip(row['user'], row['base'], row['pred'], row['gt']) if (b==gt) and (h!=gt) and (pred==gt)]), axis=1)),
        'b:nc, h:c, t:c': sum(raw_outputs.apply(lambda row: sum([1 for h, b, pred, gt in zip(row['user'], row['base'], row['pred'], row['gt']) if (b!=gt) and (h==gt) and (pred==gt)]), axis=1)),
        'b:c, h:c, t:c': sum(raw_outputs.apply(lambda row: sum([1 for h, b, pred, gt in zip(row['user'], row['base'], row['pred'], row['gt']) if (b==gt) and (h==gt) and (pred==gt)]), axis=1)),
        'b:nc, h:nc, t:c': sum(raw_outputs.apply(lambda row: sum([1 for h, b, pred, gt in zip(row['user'], row['base'], row['pred'], row['gt']) if (b!=gt) and (h!=gt) and (pred==gt)]), axis=1)),

        'b:c, h:nc, t:nc': sum(raw_outputs.apply(lambda row: sum([1 for h, b, pred, gt in zip(row['user'], row['base'], row['pred'], row['gt']) if (b==gt) and (h!=gt) and (pred!=gt)]), axis=1)),
        'b:nc, h:c, t:nc': sum(raw_outputs.apply(lambda row: sum([1 for h, b, pred, gt in zip(row['user'], row['base'], row['pred'], row['gt']) if (b!=gt) and (h==gt) and (pred!=gt)]), axis=1)),
        'b:c, h:c, t:nc': sum(raw_outputs.apply(lambda row: sum([1 for h, b, pred, gt in zip(row['user'], row['base'], row['pred'], row['gt']) if (b==gt) and (h==gt) and (pred!=gt)]), axis=1)),
        'b:nc, h:nc, t:nc': sum(raw_outputs.apply(lambda row: sum([1 for h, b, pred, gt in zip(row['user'], row['base'], row['pred'], row['gt']) if (b!=gt) and (h!=gt) and (pred!=gt)]), axis=1))
    }
    return results