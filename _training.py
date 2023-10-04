import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy 


# -------------------------
# -  Training procedures  -
# -------------------------
def train_vae(model, opt, train_loader, val_loader, num_epochs, beta=1, warmup_epochs=15, patience=3, dev="cpu", print_epoch=1):
    loss_history = {'train_loss': [], 'recon_loss': [], 'kl_loss': [], 'val_loss': []}
    
    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        kl_weight = min(beta, 1e-5 * warmup_epochs + (1 - 1e-5 * warmup_epochs) * (epoch / warmup_epochs))

        for _, _, _, data in train_loader:
            data = data.to(dev)
            opt.zero_grad()
            x_hat, mu, log_var = model(data)
            recon_loss, kl_div = model.loss(data, x_hat, mu, log_var)
            loss = recon_loss + kl_weight * kl_div
            loss.backward()
            opt.step()

            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            running_kl_loss += kl_div.item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_recon_loss = running_recon_loss / len(train_loader.dataset)
        epoch_kl_loss = running_kl_loss / len(train_loader.dataset)
        loss_history['train_loss'].append(epoch_loss)
        loss_history['recon_loss'].append(epoch_recon_loss)
        loss_history['kl_loss'].append(epoch_kl_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, _, _, data in val_loader:
                data = data.to(dev)
                x_hat, mu, log_var = model(data)
                recon_loss, kl_div = model.loss(data, x_hat, mu, log_var)
                loss = recon_loss + kl_weight * kl_div
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        loss_history['val_loss'].append(epoch_val_loss)

        if epoch % print_epoch == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss}, Recon Loss: {epoch_recon_loss}, KL Loss: {epoch_kl_loss}, Val Loss: {epoch_val_loss}")

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    
    model.load_state_dict(best_model)
    print("Best epoch:", best_epoch, "val loss:", best_val_loss)
    return model, loss_history


def train_nbvae(model, opt, train_loader, val_loader, num_epochs, beta=1, warmup_epochs=15, patience=3, dev="cpu", print_epoch=1):
    loss_history = {'train_loss': [], 'recon_loss': [], 'kl_loss': [], 'val_loss': []}
    
    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        kl_weight = min(beta, 1e-5 * warmup_epochs + (1 - 1e-5 * warmup_epochs) * (epoch / warmup_epochs))

        for data, scaling, _, _ in train_loader:
            data = data.to(dev)
            scaling = scaling.to(dev)
            opt.zero_grad()
            x_hat, mu, log_var = model(data)
            # check if x_hat, mu or log_var contains nan
            recon_loss, kl_div = model.loss(data, x_hat, mu, log_var, scaling)
            loss = recon_loss + kl_weight * kl_div
            loss.backward()
            opt.step()

            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            running_kl_loss += kl_div.item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_recon_loss = running_recon_loss / len(train_loader.dataset)
        epoch_kl_loss = running_kl_loss / len(train_loader.dataset)
        loss_history['train_loss'].append(epoch_loss)
        loss_history['recon_loss'].append(epoch_recon_loss)
        loss_history['kl_loss'].append(epoch_kl_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, scaling, _, _ in val_loader:
                data = data.to(dev)
                scaling = scaling.to(dev)
                x_hat, mu, log_var = model(data)
                recon_loss, kl_div = model.loss(data, x_hat, mu, log_var, scaling)
                loss = recon_loss + kl_weight * kl_div
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        loss_history['val_loss'].append(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

        if epoch % print_epoch == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss}, Recon Loss: {epoch_recon_loss}, KL Loss: {epoch_kl_loss}, Val Loss: {epoch_val_loss}")
    
    model.load_state_dict(best_model)
    print("Best epoch:", best_epoch, "val loss:", best_val_loss)
    return model, loss_history


def train_nbvae_embryotime(model, opt, train_loader, val_loader, num_epochs, beta=1, warmup_epochs=15, patience=3, dev="cpu", print_epoch=1, label_weight=1.0):
    loss_history = {'train_loss': [], 'recon_loss': [], 'kl_loss': [], 'val_loss': [], 'l1_loss': []}
    
    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        running_l1_loss = 0.0
        kl_weight = min(beta, 1e-5 * warmup_epochs + (1 - 1e-5 * warmup_epochs) * (epoch / warmup_epochs))

        for data, scaling, _, _, time_raw in train_loader:
            data = data.to(dev)
            scaling = scaling.to(dev)
            time_raw = time_raw.to(dev)
            mask = (time_raw != -1).float()  # create a mask for elements not equal to -1
            
            opt.zero_grad()
            x_hat, mu, log_var = model(data)
            label_preds = x_hat[:,-1]
            
            recon_loss, kl_div = model.loss(data[:,:-1], x_hat[:,:-1], mu, log_var, scaling)
            l1_loss = torch.sum(torch.abs(time_raw - label_preds) * mask) * label_weight
            running_l1_loss += l1_loss.item()
            
            total_loss = recon_loss + kl_weight * kl_div + l1_loss
            total_loss.backward()
            opt.step()

            running_loss += total_loss.item()
            running_recon_loss += recon_loss.item()
            running_kl_loss += kl_div.item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_recon_loss = running_recon_loss / len(train_loader.dataset)
        epoch_kl_loss = running_kl_loss / len(train_loader.dataset)
        epoch_l1_loss = running_l1_loss / len(train_loader.dataset)
        loss_history['train_loss'].append(epoch_loss)
        loss_history['recon_loss'].append(epoch_recon_loss)
        loss_history['kl_loss'].append(epoch_kl_loss)
        loss_history['l1_loss'].append(epoch_l1_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, scaling, _, _, time_raw in val_loader:
                data = data.to(dev)
                scaling = scaling.to(dev)
                time_raw = time_raw.to(dev)
                mask = (time_raw != -1).float()

                x_hat, mu, log_var = model(data)
                label_preds = x_hat[:,-1]
                
                recon_loss, kl_div = model.loss(data[:,:-1], x_hat[:,:-1], mu, log_var, scaling)
                l1_loss = torch.sum(torch.abs(time_raw - label_preds) * mask) * label_weight                
                total_loss = recon_loss + kl_weight * kl_div + l1_loss
                val_loss += total_loss.item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        loss_history['val_loss'].append(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

        if epoch % print_epoch == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss:.4f}, Recon Loss: {epoch_recon_loss:.4f}, KL Loss: {epoch_kl_loss:.4f}, L1 Loss: {epoch_l1_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
    
    model.load_state_dict(best_model)
    print("Best epoch:", best_epoch, "val loss:", best_val_loss)
    return model, loss_history


def train_nbvae_ctx(model, opt, train_loader, val_loader, num_epochs, beta=1, ce_weight=100, warmup_epochs=15, patience=3, dev="cpu", print_epoch=1):
    loss_history = {'train_loss': [], 'recon_loss': [], 'kl_loss': [], "ce_loss":[], 'val_loss': [], "val_ce_loss":[]}
    
    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        running_ce_loss = 0.0
        kl_weight = min(beta, 1e-5 * warmup_epochs + (1 - 1e-5 * warmup_epochs) * (epoch / warmup_epochs))

        for data, scaling, _, label in train_loader:
            data = data.to(dev)
            scaling = scaling.to(dev)
            label = label.to(dev)
            mask = (label != -1).float()  # create a mask for elements not equal to -1

            opt.zero_grad()
            x_hat, mu, log_var = model(data)
            label_preds = F.sigmoid(x_hat[:,-1].reshape((-1,1)))
            recon_loss, kl_div = model.loss(data[:,:-1], x_hat[:,:-1], mu, log_var, scaling)
            ce_loss = F.binary_cross_entropy(label_preds, label.reshape((-1,1)), weight=mask.reshape((-1,1)), reduction="sum")
            
            loss = recon_loss + kl_weight * kl_div + ce_weight * ce_loss
            loss.backward()
            opt.step()

            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            running_kl_loss += kl_div.item()
            running_ce_loss += ce_loss.item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_recon_loss = running_recon_loss / len(train_loader.dataset)
        epoch_kl_loss = running_kl_loss / len(train_loader.dataset)
        epoch_ce_loss = running_ce_loss / len(train_loader.dataset)
        loss_history['train_loss'].append(epoch_loss)
        loss_history['recon_loss'].append(epoch_recon_loss)
        loss_history['kl_loss'].append(epoch_kl_loss)
        loss_history['ce_loss'].append(epoch_ce_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_ce_loss = 0.0
        with torch.no_grad():
            for data, scaling, _, label in val_loader:
                data = data.to(dev)
                scaling = scaling.to(dev)
                label = label.to(dev)
                mask = (label != -1).float()  # create a mask for elements not equal to -1

                opt.zero_grad()
                x_hat, mu, log_var = model(data)
                label_preds = F.sigmoid(x_hat[:,-1].reshape((-1,1)))
                recon_loss, kl_div = model.loss(data[:,:-1], x_hat[:,:-1], mu, log_var, scaling)
                ce_loss = F.binary_cross_entropy(label_preds, label.reshape((-1,1)), weight=mask.reshape((-1,1)), reduction="sum")
                
                val_loss += recon_loss + kl_weight * kl_div + ce_weight * ce_loss
                val_ce_loss += ce_loss

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_ce_loss = val_ce_loss / len(val_loader.dataset)
        loss_history['val_loss'].append(epoch_val_loss)
        loss_history['val_ce_loss'].append(epoch_val_ce_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

        if epoch % print_epoch == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss:.4f}, Recon Loss: {epoch_recon_loss:.4f}, KL Loss: {epoch_kl_loss:.4f}, CE Loss: {epoch_ce_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val CE Loss: {epoch_val_ce_loss:.4f}")
    model.load_state_dict(best_model)
    print("Best epoch:", best_epoch, "val loss:", best_val_loss)
    return model, loss_history

