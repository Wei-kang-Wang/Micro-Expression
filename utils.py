import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import accuracy_score

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn



def metric_acc(output, y_true):
    output = output.detach().numpy()
    y_true = y_true.detach().numpy()
    y_pred = output.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    return acc



def plot_lr_find(lr_list, loss_list):
    lr_list, loss_list = lr_list[10:], loss_list[10:]

    begin_index = int(len(loss_list) / 3)
    min_index = np.argmin(loss_list[begin_index:])
    max_val = np.max(loss_list[:min_index])

    plot_loss_list = [x for x in loss_list if x <= max_val]
    plot_lr_list = [lr_list[i] for i, x in enumerate(loss_list) if x <= max_val]

    fig, ax = plt.subplots()
    ax.plot(plot_lr_list[:-1], plot_loss_list[:-1])
    ax.set_xscale('log')



def lr_find(model, data, train_option, device, lr_init=1e-6, beta=0.98):

    opt = train_option['opt_class'](
        filter( lambda p: p.requires_grad, model.parameters() ),
        lr=lr_init, weight_decay=train_option['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 1.1 ** epoch)

    iter_num, avg_loss, best_loss = 0, 0, float('inf')
    lr_list, loss_list = [], []
    
    os.makedirs('tmp', exist_ok=True)
    torch.save(model.state_dict(), 'tmp/lr_find_before.pkl')
    model.train()

    while True:
        iter_num += 1
        scheduler.step()
        cur_lr = opt.param_groups[0]['lr']
        lr_list.append(cur_lr)

        for X_batch, y_batch in data['train']:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            opt.zero_grad()
            out = model(X_batch)
            loss = train_option['criterion'](out, y_batch)
            loss.backward()
            opt.step()

            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            corr_avg_loss = avg_loss / (1 - beta ** iter_num)
            loss_list.append(corr_avg_loss)
            best_loss = min(corr_avg_loss, best_loss)
            break


        if (iter_num - 1) % 25 == 0:
            print( 'iter_num=%d, cur_lr=%g, loss=%g' % (iter_num - 1, cur_lr, loss_list[-1]) )
        
            
        if iter_num > 10 and corr_avg_loss > 4 * best_loss:
            break


    model.load_state_dict( torch.load('tmp/lr_find_before.pkl') )
    return lr_list, loss_list




def model_predict(model, data_loader, view, device):
    model.eval()
    y_true, output = [], []

    with torch.no_grad():
        for i, (XA_batch, XB_batch, y_batch) in enumerate(data_loader):
            if view == 'occ':
                cur_batch = XA_batch.to(device)
            elif view == 'clean':
                cur_batch = XB_batch.to(device)            

            out = model(cur_batch)
            output.append( out.to('cpu') )
            y_true.append(y_batch)

    y_true, output = torch.cat(y_true), torch.cat(output)
    return y_true, output




def model_predict_AB(modelA, modelB, data_loader, device):
    modelA.eval(), modelB.eval()
    y_true, outputA, outputB = [], [], []

    with torch.no_grad():
        for i, (XA_batch, XB_batch, y_batch) in enumerate(data_loader):
            XA_batch, XB_batch = XA_batch.to(device), XB_batch.to(device)

            outA, outB = modelA(XA_batch), modelB(XB_batch)
            outputA.append( outA.to('cpu') )
            outputB.append( outB.to('cpu') )
            y_true.append(y_batch)

    y_true, outputA, outputB = torch.cat(y_true), torch.cat(outputA), torch.cat(outputB)
    return y_true, outputA, outputB



def LIR_loss(lossA,lossB):
    temp = lossB-lossA
    c = temp>0
    loss = temp[c].sum()/temp.size(0)
    return loss

    
    
def model_fit(modelA, modelB, modelD1,modelD2,lr, lrD1,lrD2,max_epoch, data, train_option, device, print_interval=100):
    
    opt = train_option['opt_class'](
        [
            {'params': filter( lambda p: p.requires_grad, modelA.parameters() )}
        ],
        lr=lr, weight_decay=train_option['weight_decay']
    )
    optD1 = train_option['opt_class'](modelD1.parameters(), lr=lrD1, weight_decay=train_option['weight_decay'])
    optD2 = train_option['opt_class'](modelD2.parameters(), lr=lrD2, weight_decay=train_option['weight_decay'])
    criterion_BCE = nn.BCELoss()
    
    best_accA = -1

    for epoch in range(max_epoch):

        t0, t1 = time(), time()
        cur_lr = opt.param_groups[0]['lr']
        print( 'Epoch=%d, lr=%g' % (epoch, cur_lr) )

        modelA.train()
        modelD1.train()
        modelD2.train()
        modelB.eval()
        y_train, outputA_train, outputB_train = [], [], []

        for batch_i, (XA_batch, XB_batch, y_batch) in enumerate(data['train']):
            XA_batch, XB_batch, y_batch = XA_batch.to(device), XB_batch.to(device), y_batch.to(device)

            label_one = torch.ones(XA_batch.size(0), device=device)  # 1 for ViewA
            label_zero = torch.zeros(XB_batch.size(0), device=device)  # 0 for ViewB

            
            
            # ---------------- Training modelD ----------------
            optD1.zero_grad()
            optD2.zero_grad()
            
            midA,highA = modelA.partial_forward_mid_high(XA_batch)
            midB,highB = modelB.partial_forward_mid_high(XB_batch)
            # real label=1 for view A

            output_real1 = modelD1(midA)
            loss_real1 = criterion_BCE(output_real1.squeeze(), label_one)
            acc_real1 = (output_real1 >= 0.5).data.float().mean()

            # fake label=0 for view B
            
            output_fake1 = modelD1(midB)
            loss_fake1 = criterion_BCE(output_fake1.squeeze(), label_zero)
            acc_fake1 = (output_fake1 < 0.5).data.float().mean()

            # 汇总real和fake
            lossD1 = loss_real1 + loss_fake1
            accD1 = (acc_real1 + acc_fake1) / 2

            lossD1.backward(retain_graph=True)
            optD1.step()
            
            
            
            
            output_real2 = modelD2(highA)
            loss_real2 = criterion_BCE(output_real2.squeeze(), label_one)
            acc_real2 = (output_real2 >= 0.5).data.float().mean()

            # fake label=0 for view B
            
            output_fake2 = modelD2(highB)
            loss_fake2 = criterion_BCE(output_fake2.squeeze(), label_zero)
            acc_fake2 = (output_fake2 < 0.5).data.float().mean()

            # 汇总real和fake
            lossD2 = loss_real2 + loss_fake2
            accD2 = (acc_real2 + acc_fake2) / 2

            lossD2.backward()
            optD2.step()
            
            

            
            # ---------------- Training modelA and modelB ----------------
            opt.zero_grad()
            
            # 前向传播
            midA,highA, outA = modelA.forward_with_mid_high(XA_batch)
            midB,highB, outB = modelB.forward_with_mid_high(XB_batch)

            term1 = train_option['criterion'](outA, y_batch)
            term2 = train_option['criterion'](outB, y_batch)
            term4 = LIR_loss(term1,term2) * train_option['C4']
            term3 = F.mse_loss(outA, outB) * train_option['C3'] # 欧式距离
            
            # real label=0 for view A（打上相反的标签）
            term1 = term1.mean()
            term2 = term2.mean() * train_option['C2']
            
            
            output_real1 = modelD1(midA)
            loss_real1 = criterion_BCE(output_real1.squeeze(), label_zero)
            cheat_real1 = (output_real1 <= 0.5).data.float().mean() # 对于featureA，对应标签1，要让modelD的输出值小于0.5才算欺骗成功

            # fake label=1 for view B（打上相反的标签）
            output_fake1 = modelD1(midB)
            loss_fake1 = criterion_BCE(output_fake1.squeeze(), label_one)
            cheat_fake1 = (output_fake1 > 0.5).data.float().mean()
            cheat1 = (cheat_real1 + cheat_fake1) / 2          
            
            
            output_real2 = modelD2(highA)
            loss_real2 = criterion_BCE(output_real2.squeeze(), label_zero)
            cheat_real2 = (output_real2 <= 0.5).data.float().mean() # 对于featureA，对应标签1，要让modelD的输出值小于0.5才算欺骗成功

            # fake label=1 for view B（打上相反的标签）
            output_fake2 = modelD2(highB)
            loss_fake2 = criterion_BCE(output_fake2.squeeze(), label_one)
            cheat_fake2 = (output_fake2 > 0.5).data.float().mean()
            cheat2 = (cheat_real2 + cheat_fake2) / 2
            
            
            
            # 汇总
            term5 = (loss_real1 + loss_fake1) * train_option['C5']
            term6 = (loss_real2 + loss_fake2) * train_option['C6']
            
            loss = term1 + term2 + term3 + term4 + term5 + term6

            
            #loss = term1 + term2 + term5 + term6
            loss.backward()
            opt.step()

            y_train.append( y_batch.to('cpu') )
            outputA_train.append( outA.to('cpu') )
            outputB_train.append( outB.to('cpu') )

            if batch_i % print_interval == 0:
                print('\tbatch_i=%d\n\t\t\tloss=%f[%f, %f, %f, %f, %f, %f]\tin %.2f s' %
                        (batch_i,
                        loss.item(), term1.item(), term2.item(),term3.item(),term4.item(),term5.item(),term6.item(),time() - t1))
                t1 = time()

            
            
        y_train, outputA_train, outputB_train = torch.cat(y_train), torch.cat(outputA_train), torch.cat(outputB_train)
        lossA_train = train_option['criterion_cpu'](outputA_train, y_train).item()
        lossB_train = train_option['criterion_cpu'](outputB_train, y_train).item()
        accA_train = metric_acc(outputA_train, y_train)
        accB_train = metric_acc(outputB_train, y_train)
        print('Train\tloss=[%f, %f]\tacc=[%f, %f]\tin %.2f s' %
                (lossA_train, lossB_train, accA_train, accB_train, time() - t0))


        if 'valid' in data:
            t0 = time()
            y_valid, outputA_valid, outputB_valid = model_predict_AB(modelA, modelB, data['valid'], device)
            lossA_valid = train_option['criterion_cpu'](outputA_valid, y_valid).item()
            lossB_valid = train_option['criterion_cpu'](outputB_valid, y_valid).item()
            accA_valid = metric_acc(outputA_valid, y_valid)
            accB_valid = metric_acc(outputB_valid, y_valid)
            
            ending = '\tBetter!' if accA_valid > best_accA else ''
            if accA_valid > best_accA:
                torch.save(modelA.state_dict(), ('occ_%.4f.pkl' % accA_valid))
            print('Valid\tloss=[%f, %f]\tacc=[%f, %f]\tin %.2f s%s' %
                   (lossA_valid, lossB_valid, accA_valid, accB_valid, time() - t0, ending))
            best_accA = max(accA_valid, best_accA)


        if 'test' in data:
            t0 = time()
            y_test, outputA_test, outputB_test = model_predict_AB(modelA, modelB, data['test'], device)
            lossA_test = train_option['criterion_cpu'](outputA_test, y_test).item()
            lossB_test = train_option['criterion_cpu'](outputB_test, y_test).item()
            accA_test = metric_acc(outputA_test, y_test)
            accB_test = metric_acc(outputB_test, y_test)
            
            if accA_test> best_accA:
                torch.save(modelA.state_dict(), ('occ_%.4f.pkl' % accA_test))
            
            ending = '\tBetter!' if 'valid' not in data and accA_test > best_accA else ''
            print('Test\tloss=[%f, %f]\tacc=[%f, %f]\tin %.2f s%s' %
                   (lossA_test, lossB_test, accA_test, accB_test, time() - t0, ending))
            if 'valid' not in data:
                best_accA = max(accA_test, best_accA)
        
        print() # end epoch

    print('best_accA = %g' % best_accA)