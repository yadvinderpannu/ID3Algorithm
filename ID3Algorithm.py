import numpy as np
import pandas as pd
import sys

# Function to split dataset as 80% for training and 20% test
def split_data(data,attr):

    class_list = data[attr[-1]].unique().tolist()

    train = pd.DataFrame(columns=data.columns)
    test = pd.DataFrame(columns=data.columns)

    for cl in class_list:
        cl_data = data.loc[data[attr[-1]] == cl]
        s = round(cl_data.shape[0]*0.8)
        ind = np.random.choice(cl_data.index.values, s, replace=False).tolist()
        indexs = cl_data.index.tolist()
        for v in ind:
            indexs.remove(v)
        train = train.append(cl_data.loc[ind], ignore_index=True)
        test = test.append(cl_data.loc[indexs], ignore_index=True)

    return [train, test]

# Function for determining K-fold CV split
def k_fold_split(k, data, attr):

    train = []
    test = []
    indexs = data.index

    for i in range(k):
        ind = np.random.choice(indexs,int(data.shape[0]/k),replace=False).tolist()
        indexs = np.delete(indexs, ind).tolist()
        train.append(np.delete(data.index, ind).tolist())
        test.append(ind)


    return train, test

# Function determines the best attribute by entropy calculation
def get_best_attribute(data, attr):

    max_gain = -1
    parent_ratio = data[attr[-1]].value_counts() / data.shape[0]
    p_ent = np.dot(-parent_ratio, np.log2(parent_ratio))
    for a in attr[:-1]:
        if str(data[a].dtype) == 'object':
            val = data[a].value_counts().index[0]
            ls = data.loc[data[a] == val]
            rs = data.loc[data[a] != val]
            ls_r = ls[attr[-1]].value_counts() / ls.shape[0]
            rs_r = rs[attr[-1]].value_counts() / rs.shape[0]
            gain = p_ent - (ls.shape[0] / data.shape[0]) * np.dot(-ls_r, np.log2(ls_r)) - (
                    rs.shape[0] / data.shape[0]) * np.dot(-rs_r, np.log2(rs_r))
            if gain > max_gain:
                max_gain = gain
                best_attr = a
                spt = val
        else:
            for th in pd.qcut(data[a], 5, retbins=True, duplicates='drop')[1]:
                ls = data[data[a] <= th]
                rs = data[data[a] > th]
                ls_r = ls[attr[-1]].value_counts() / ls.shape[0]
                rs_r = rs[attr[-1]].value_counts() / rs.shape[0]
                gain = p_ent -(ls.shape[0]/data.shape[0])*np.dot(-ls_r, np.log2(ls_r)) - (rs.shape[0]/data.shape[0])*np.dot(-rs_r, np.log2(rs_r))
                if gain > max_gain:
                    max_gain = gain
                    best_attr = a
                    spt = th

    return [best_attr, spt, max_gain]

# Function recursively determines nodes of the decision tree
def dtree_train(data, attr):

    if (data.shape[0] == 0):
        node = ""
    elif (data.shape[0] > 0 and len(attr) == 1):
        node = data[attr[-1]].value_counts(sort=True, ascending=False).index[0]
    elif(data[attr[-1]].nunique() == 1):
        node = data[attr[-1]].tolist()[0]
    else:
        [best_attr, spt, gain] = get_best_attribute(data, attr)
        attr.remove(best_attr)
        if str(data[best_attr].dtype) == 'object':
            node = {'attr': best_attr,
                    'spt_value': spt,
                    'gain': gain,
                    'l_child': dtree_train(data[data[best_attr] == spt], attr),
                    'r_child': dtree_train(data[data[best_attr] != spt], attr),
                    'type':'cat'}
        else:
            node = {'attr': best_attr,
                    'spt_value': spt,
                    'gain': gain,
                    'l_child': dtree_train(data[data[best_attr] <= spt], attr),
                    'r_child': dtree_train(data[data[best_attr] > spt], attr),
                    'type':'num'}

    return node

# Function to predict the class for the test data using the obtained decision tree
def predict(tree, test_data):

    pred = list()
    accuracy = 0

    for ix, tx in test_data.iterrows():
        node = tree
        while True:
            if node['type'] is 'num':
                if tx[node['attr']] <= node['spt_value']:
                    n = node['l_child']
                else:
                    n = node['r_child']
            else:
                if tx[node['attr']] is node['spt_value']:
                    n = node['l_child']
                else:
                    n = node['r_child']

            if type(n) is str:
                break
            node = n

        pred.append(n)
        if n is tx[-1]:
            accuracy += 1

    print("\nTest Data Prediction -")
    print(pred)
    print("Accuracy=%f"%(accuracy/test_data.shape[0]))
    return (accuracy/test_data.shape[0])

# Function displays the obtained decision tree
def print_tree(tree):

    q = [tree]
    i = 0

    while True:

        if q[i]['attr'] is "Leaf":
            if q[i]['class'] is not "":
                print("Node=%3d\tAttribute=%s\tClass=%s"%(i+1,q[i]['attr'],q[i]['class']))
        else:
            if q[i]['type'] is 'num':
                print("Node=%3d\tAttribute=%s\tSplit point=%5.2f\tGain=%f" % (
            i+1, q[i]['attr'], q[i]['spt_value'], q[i]['gain']))
            else:
                print("Node=%3d\tAttribute=%s\tSplit point=%s\tGain=%f" % (
                    i + 1, q[i]['attr'], q[i]['spt_value'], q[i]['gain']))

            if type(q[i]['l_child']) is dict:
                q.append(q[i]['l_child'])
            else:
                q.append({"attr":"Leaf","class":q[i]['l_child']})

            if type(q[i]['r_child']) is dict:
                q.append(q[i]['r_child'])
            else:
                q.append({"attr":"Leaf","class":q[i]['r_child']})

        i += 1
        if i == len(q):
            break

    return


# Main execution starts here
if __name__ == "__main__":
    if len(sys.argv) > 2:
        attr_file = sys.argv[1]
        data_file = sys.argv[2]
    else:
        print("Please provide the path to the attribute file and data file")
        sys.exit()


    with open(attr_file.strip(), 'r') as fh:
        attrs = [str(atr.strip()) for atr in fh.read().split(",")]
        fh.close()

    data = pd.read_csv(data_file, header=None, names=attrs)
    split_option = input("Select the implementation type\n1) Dataset Split\n2)K-Fold CV\n")

    if split_option == "1":
        [train_data, test_data] = split_data(data,attrs)
        dt = dtree_train(train_data, attrs)
        print("\nDecision tree -\n")
        print_tree(dt)
        accu = predict(dt, test_data)

    elif split_option == "2":
        kfold = int(input("Enter 'K' value-\n"))
        accu = list()
        trindex, teindex = k_fold_split(kfold,data,attrs)
        for count,(i,j) in enumerate(zip(trindex,teindex)):
            train_data = data.iloc[i]
            test_data = data.iloc[j]
            dt = dtree_train(train_data, list(attrs))
            print("\nDecision Tree for K = %d-\n"%(count+1))
            print_tree(dt)
            accu.append(predict(dt, test_data))

        print("\n Average accuracy=%f"%(np.sum(np.array(accu))/kfold))

    else:
        print("Select correct option")
        sys.exit()

# yadvinderpannu

