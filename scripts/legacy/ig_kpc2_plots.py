=== Found in Copia_de_Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil_experimentos) (1).ipynb Cell 23 ===
import torch
import numpy as np
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

def analizar_interpretabilidad_mil(model, tokenizer, bag, device="cuda"):
    model.to(device)
    model.eval()

    sequences = bag["sequences"]
    genome_id = bag.get("genome_id", "Unknown")
    label = bag.get("label", "?")

    print(f"🔍 Analizando Genoma: {genome_id} (Label: {label})")

    # --- PASO 1: Identificar el Gen con mayor Probabilidad (Top-1) ---
    all_probs = []
    sub_batch = 8

    with torch.no_grad():
        for start in range(0, len(sequences), sub_batch):
            chunk = sequences[start : start + sub_batch]
            inputs = tokenizer(chunk, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
            sp_ids = torch.full((inputs['input_ids'].size(0),), 6, device=device)

            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'), species_ids=sp_ids)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs)

    all_probs = torch.cat(all_probs)
    top_instance_idx = torch.argmax(all_probs).item()
    top_sequence = sequences[top_instance_idx]

    print(f"📍 Gen detectado como 'Driver' de Resistencia: {top_sequence[:60]}...")
    print(f"🔥 Probabilidad del Gen: {all_probs[top_instance_idx].item():.4f}")

    # --- PASO 2: Integrated Gradients sobre el Gen Ganador ---
    def predict_wrapper(inputs_embeds):
        # FIX: Expand species_ids to match the batch size created by Captum steps
        batch_size = inputs_embeds.shape[0]
        sp_id = torch.full((batch_size,), 6, device=device)

        outputs = model.ntv3(inputs_embeds=inputs_embeds, species_ids=sp_id, output_hidden_states=True)
        last_hidden_layer = outputs.hidden_states[-1]
        cls_output = last_hidden_layer[:, 0, :]
        logits = model.classifier(model.dropout(cls_output))
        return logits[:, 1]

    ig = IntegratedGradients(predict_wrapper)

    inputs_top = tokenizer(top_sequence, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs_top.input_ids

    with torch.no_grad():
        input_embeds = model.ntv3.core.embed_layer(input_ids)

    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    baseline_embeds = model.ntv3.core.embed_layer(baseline_ids)

    # Run attribution
    attributions = ig.attribute(inputs=input_embeds, baselines=baseline_embeds, target=None)

    scores = attributions.squeeze(0).norm(p=2, dim=-1).cpu().detach().numpy()

    # --- PASO 3: Visualización ---
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(scores)), scores, color='crimson', alpha=0.7)
    plt.title(f"Importancia (Integrated Gradients) - Gen Driver en {genome_id}")
    plt.xlabel("Posición del Token / Nucleótido")
    plt.ylabel("Atribución")
    plt.grid(alpha=0.2)
    plt.show()

    return top_sequence, scores, tokenizer.convert_ids_to_tokens(input_ids[0])
============================================================
=== Found in Copia_de_Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil_experimentos) (1).ipynb Cell 33 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        # CORRECCIÓN: Usar core.embed_layer en lugar de embeddings
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated = interpolated.detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    # ── MAPEAR ATRIBUCIONES A KMERS DE LA SECUENCIA ORIGINAL ─
    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)

    # Construir kmers con sus atribuciones
    kmer_scores = {}
    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        kmer_seq = seq_limpia[i:i+kmer]
        score = float(np.mean(token_importance[i:i+kmer]))
        if kmer_seq in kmer_scores:
            kmer_scores[kmer_seq] += score
        else:
            kmer_scores[kmer_seq] = score

    # Top 20 kmers por score total
    top_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\nTop 10 motivos 6-mer ({label_str}):")
    for km, sc in top_kmers[:10]:
        print(f"  {km}: {sc:.6f}")

    # ── FIGURA ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    nombres = [k[0] for k in top_kmers]
    valores = [k[1] for k in top_kmers]
    color = 'red' if label_str == 'Resistant' else 'blue'
    plt.barh(nombres, valores, color=color, alpha=0.85)
    plt.xlabel("Integrated Gradients Attribution (mean over k-mer positions)")
    plt.title(f"Figure — Integrated Gradients 6-mer Motifs: {label_str} sequence\n"
              f"(top 20 motifs by attribution score)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ruta = f'{RUTA_BASE}/ig_kmers_{label_str.lower()}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Guardado: ig_kmers_{label_str.lower()}.png")
    return top_kmers

# ── CORRER ───────────────────────────────────────────────────
idx_R = np.where(labels_muestra == 1)[0][0]
idx_S = np.where(labels_muestra == 0)[0][0]

print("🧬 Integrated Gradients 6-mer — Resistente...")
motivos_R = integrated_gradients_motifs(secuencias[idx_R], "Resistant", target_class=1)

print("\n🧬 Integrated Gradients 6-mer — Susceptible...")
motivos_S = integrated_gradients_motifs(secuencias[idx_S], "Susceptible", target_class=0)

============================================================
=== Found in Copia_de_Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil_experimentos) (1).ipynb Cell 34 ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Resistente
nombres_R = [k[0] for k in motivos_R[:15]]
valores_R = [k[1] for k in motivos_R[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("Resistant sequence\nTop 15 6-mer motifs", fontsize=12)

# Susceptible
nombres_S = [k[0] for k in motivos_S[:15]]
valores_S = [k[1] for k in motivos_S[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle("Figure X — Integrated Gradients 6-mer Motif Analysis\n"
             "NTv3 attribution scores for Ertapenem Resistance vs Susceptibility",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_IG_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X guardada")
============================================================
=== Found in Copia_de_Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil_experimentos) (1).ipynb Cell 45 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask",
                    (input_ids != tokenizer.pad_token_id).long()).to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = (baseline_embeds + alpha *
                       (input_embeds - baseline_embeds)).detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads *
                   (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)
    kmer_scores = {}

    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        motivo = seq_limpia[i:i+kmer]
        if len(motivo) == kmer and all(c in 'ATGC' for c in motivo):
            score = float(np.mean(token_importance[i:i+kmer]))
            if motivo in kmer_scores:
                kmer_scores[motivo] = max(kmer_scores[motivo], score)
            else:
                kmer_scores[motivo] = score

    return sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("✅ Función integrated_gradients_motifs definida")
============================================================
=== Found in Copia_de_Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil_experimentos).ipynb Cell 23 ===
import torch
import numpy as np
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

def analizar_interpretabilidad_mil(model, tokenizer, bag, device="cuda"):
    model.to(device)
    model.eval()

    sequences = bag["sequences"]
    genome_id = bag.get("genome_id", "Unknown")
    label = bag.get("label", "?")

    print(f"🔍 Analizando Genoma: {genome_id} (Label: {label})")

    # --- PASO 1: Identificar el Gen con mayor Probabilidad (Top-1) ---
    all_probs = []
    sub_batch = 8

    with torch.no_grad():
        for start in range(0, len(sequences), sub_batch):
            chunk = sequences[start : start + sub_batch]
            inputs = tokenizer(chunk, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
            sp_ids = torch.full((inputs['input_ids'].size(0),), 6, device=device)

            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'), species_ids=sp_ids)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs)

    all_probs = torch.cat(all_probs)
    top_instance_idx = torch.argmax(all_probs).item()
    top_sequence = sequences[top_instance_idx]

    print(f"📍 Gen detectado como 'Driver' de Resistencia: {top_sequence[:60]}...")
    print(f"🔥 Probabilidad del Gen: {all_probs[top_instance_idx].item():.4f}")

    # --- PASO 2: Integrated Gradients sobre el Gen Ganador ---
    def predict_wrapper(inputs_embeds):
        # FIX: Expand species_ids to match the batch size created by Captum steps
        batch_size = inputs_embeds.shape[0]
        sp_id = torch.full((batch_size,), 6, device=device)

        outputs = model.ntv3(inputs_embeds=inputs_embeds, species_ids=sp_id, output_hidden_states=True)
        last_hidden_layer = outputs.hidden_states[-1]
        cls_output = last_hidden_layer[:, 0, :]
        logits = model.classifier(model.dropout(cls_output))
        return logits[:, 1]

    ig = IntegratedGradients(predict_wrapper)

    inputs_top = tokenizer(top_sequence, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs_top.input_ids

    with torch.no_grad():
        input_embeds = model.ntv3.core.embed_layer(input_ids)

    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    baseline_embeds = model.ntv3.core.embed_layer(baseline_ids)

    # Run attribution
    attributions = ig.attribute(inputs=input_embeds, baselines=baseline_embeds, target=None)

    scores = attributions.squeeze(0).norm(p=2, dim=-1).cpu().detach().numpy()

    # --- PASO 3: Visualización ---
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(scores)), scores, color='crimson', alpha=0.7)
    plt.title(f"Importancia (Integrated Gradients) - Gen Driver en {genome_id}")
    plt.xlabel("Posición del Token / Nucleótido")
    plt.ylabel("Atribución")
    plt.grid(alpha=0.2)
    plt.show()

    return top_sequence, scores, tokenizer.convert_ids_to_tokens(input_ids[0])
============================================================
=== Found in Copia_de_Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil_experimentos).ipynb Cell 33 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        # CORRECCIÓN: Usar core.embed_layer en lugar de embeddings
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated = interpolated.detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    # ── MAPEAR ATRIBUCIONES A KMERS DE LA SECUENCIA ORIGINAL ─
    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)

    # Construir kmers con sus atribuciones
    kmer_scores = {}
    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        kmer_seq = seq_limpia[i:i+kmer]
        score = float(np.mean(token_importance[i:i+kmer]))
        if kmer_seq in kmer_scores:
            kmer_scores[kmer_seq] += score
        else:
            kmer_scores[kmer_seq] = score

    # Top 20 kmers por score total
    top_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\nTop 10 motivos 6-mer ({label_str}):")
    for km, sc in top_kmers[:10]:
        print(f"  {km}: {sc:.6f}")

    # ── FIGURA ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    nombres = [k[0] for k in top_kmers]
    valores = [k[1] for k in top_kmers]
    color = 'red' if label_str == 'Resistant' else 'blue'
    plt.barh(nombres, valores, color=color, alpha=0.85)
    plt.xlabel("Integrated Gradients Attribution (mean over k-mer positions)")
    plt.title(f"Figure — Integrated Gradients 6-mer Motifs: {label_str} sequence\n"
              f"(top 20 motifs by attribution score)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ruta = f'{RUTA_BASE}/ig_kmers_{label_str.lower()}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Guardado: ig_kmers_{label_str.lower()}.png")
    return top_kmers

# ── CORRER ───────────────────────────────────────────────────
idx_R = np.where(labels_muestra == 1)[0][0]
idx_S = np.where(labels_muestra == 0)[0][0]

print("🧬 Integrated Gradients 6-mer — Resistente...")
motivos_R = integrated_gradients_motifs(secuencias[idx_R], "Resistant", target_class=1)

print("\n🧬 Integrated Gradients 6-mer — Susceptible...")
motivos_S = integrated_gradients_motifs(secuencias[idx_S], "Susceptible", target_class=0)

============================================================
=== Found in Copia_de_Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil_experimentos).ipynb Cell 34 ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Resistente
nombres_R = [k[0] for k in motivos_R[:15]]
valores_R = [k[1] for k in motivos_R[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("Resistant sequence\nTop 15 6-mer motifs", fontsize=12)

# Susceptible
nombres_S = [k[0] for k in motivos_S[:15]]
valores_S = [k[1] for k in motivos_S[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle("Figure X — Integrated Gradients 6-mer Motif Analysis\n"
             "NTv3 attribution scores for Ertapenem Resistance vs Susceptibility",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_IG_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X guardada")
============================================================
=== Found in Copia_de_Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil_experimentos).ipynb Cell 45 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask",
                    (input_ids != tokenizer.pad_token_id).long()).to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = (baseline_embeds + alpha *
                       (input_embeds - baseline_embeds)).detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads *
                   (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)
    kmer_scores = {}

    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        motivo = seq_limpia[i:i+kmer]
        if len(motivo) == kmer and all(c in 'ATGC' for c in motivo):
            score = float(np.mean(token_importance[i:i+kmer]))
            if motivo in kmer_scores:
                kmer_scores[motivo] = max(kmer_scores[motivo], score)
            else:
                kmer_scores[motivo] = score

    return sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("✅ Función integrated_gradients_motifs definida")
============================================================
=== Found in Flaming_AI(produccion) (para hacer el probar otras secuencias para ver que dice el modelo).ipynb Cell 15 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        # CORRECCIÓN: Usar core.embed_layer en lugar de embeddings
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated = interpolated.detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    # ── MAPEAR ATRIBUCIONES A KMERS DE LA SECUENCIA ORIGINAL ─
    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)

    # Construir kmers con sus atribuciones
    kmer_scores = {}
    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        kmer_seq = seq_limpia[i:i+kmer]
        score = float(np.mean(token_importance[i:i+kmer]))
        if kmer_seq in kmer_scores:
            kmer_scores[kmer_seq] += score
        else:
            kmer_scores[kmer_seq] = score

    # Top 20 kmers por score total
    top_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\nTop 10 motivos 6-mer ({label_str}):")
    for km, sc in top_kmers[:10]:
        print(f"  {km}: {sc:.6f}")

    # ── FIGURA ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    nombres = [k[0] for k in top_kmers]
    valores = [k[1] for k in top_kmers]
    color = 'red' if label_str == 'Resistant' else 'blue'
    plt.barh(nombres, valores, color=color, alpha=0.85)
    plt.xlabel("Integrated Gradients Attribution (mean over k-mer positions)")
    plt.title(f"Figure — Integrated Gradients 6-mer Motifs: {label_str} sequence\n"
              f"(top 20 motifs by attribution score)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ruta = f'{RUTA_BASE}/ig_kmers_{label_str.lower()}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Guardado: ig_kmers_{label_str.lower()}.png")
    return top_kmers

# ── CORRER ───────────────────────────────────────────────────
idx_R = np.where(labels_muestra == 1)[0][0]
idx_S = np.where(labels_muestra == 0)[0][0]

print("🧬 Integrated Gradients 6-mer — Resistente...")
motivos_R = integrated_gradients_motifs(secuencias[idx_R], "Resistant", target_class=1)

print("\n🧬 Integrated Gradients 6-mer — Susceptible...")
motivos_S = integrated_gradients_motifs(secuencias[idx_S], "Susceptible", target_class=0)

============================================================
=== Found in Flaming_AI(produccion) (para hacer el probar otras secuencias para ver que dice el modelo).ipynb Cell 16 ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Resistente
nombres_R = [k[0] for k in motivos_R[:15]]
valores_R = [k[1] for k in motivos_R[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("Resistant sequence\nTop 15 6-mer motifs", fontsize=12)

# Susceptible
nombres_S = [k[0] for k in motivos_S[:15]]
valores_S = [k[1] for k in motivos_S[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle("Figure X — Integrated Gradients 6-mer Motif Analysis\n"
             "NTv3 attribution scores for Ertapenem Resistance vs Susceptibility",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_IG_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X guardada")
============================================================
=== Found in Flaming_AI(produccion) (para hacer el probar otras secuencias para ver que dice el modelo).ipynb Cell 22 ===
# IG sobre KPC-3 real
print("🧬 Integrated Gradients — KPC-3 (carbapenemasa clínica)...")
motivos_KPC3 = integrated_gradients_motifs(seq_kpc3, "KPC-3 Resistant", target_class=1)

# Buscar también una secuencia susceptible con gen bien identificado
# para comparación limpia
seq_sus = val_df[val_df['label'] == 0].iloc[0]['sequence']
print("\n🧬 Integrated Gradients — Susceptible...")
motivos_SUS = integrated_gradients_motifs(seq_sus, "Susceptible", target_class=0)

# Figura comparativa final
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

nombres_R = [k[0] for k in motivos_KPC3[:15]]
valores_R = [k[1] for k in motivos_KPC3[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("KPC-3 beta-lactamase (Resistant)\nTop 15 6-mer motifs", fontsize=12)

nombres_S = [k[0] for k in motivos_SUS[:15]]
valores_S = [k[1] for k in motivos_SUS[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle(
    "Figure X — Integrated Gradients 6-mer Motif Analysis\n"
    "KPC-3 beta-lactamase vs Susceptible: NTv3 attribution scores for Ertapenem resistance",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_KPC3_vs_susceptible.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X final guardada: figura_X_KPC3_vs_susceptible.png")
============================================================
=== Found in Flaming_AI(produccion) (para hacer el probar otras secuencias para ver que dice el modelo).ipynb Cell 31 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask",
                    (input_ids != tokenizer.pad_token_id).long()).to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = (baseline_embeds + alpha *
                       (input_embeds - baseline_embeds)).detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads *
                   (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)
    kmer_scores = {}

    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        motivo = seq_limpia[i:i+kmer]
        if len(motivo) == kmer and all(c in 'ATGC' for c in motivo):
            score = float(np.mean(token_importance[i:i+kmer]))
            if motivo in kmer_scores:
                kmer_scores[motivo] = max(kmer_scores[motivo], score)
            else:
                kmer_scores[motivo] = score

    return sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("✅ Función integrated_gradients_motifs definida")
============================================================
=== Found in Flaming_AI(produccion) (para hacer el probar otras secuencias para ver que dice el modelo).ipynb Cell 32 ===
# ── IG SOBRE LAS 25 KPC-3 ────────────────────────────────────
print("Corriendo IG sobre 25 secuencias KPC-3...")

todos_motivos = {}

for idx, row in tqdm(df_kpc3.iterrows(), total=len(df_kpc3), desc="IG KPC-3"):
    try:
        motivos = integrated_gradients_motifs(
            row['sequence'],
            label_str="KPC-3",
            target_class=1,
            steps=50,
            kmer=6
        )
        for motivo, score in motivos:
            if motivo not in todos_motivos:
                todos_motivos[motivo] = []
            todos_motivos[motivo].append(score)
    except Exception as e:
        print(f"Error en secuencia {idx}: {e}")
        continue

# ── AGREGAR POR MOTIVO ───────────────────────────────────────
motivos_agregados = {
    motivo: np.mean(scores)
    for motivo, scores in todos_motivos.items()
    if len(scores) >= 5  # aparece en al menos 5 de 25 secuencias
}

top_motivos = sorted(motivos_agregados.items(), key=lambda x: x[1], reverse=True)[:20]

print(f"\n✅ Motivos únicos encontrados: {len(todos_motivos)}")
print(f"Motivos consistentes (≥5 secuencias): {len(motivos_agregados)}")
print(f"\nTop 10 motivos universales KPC-3:")
for motivo, score in top_motivos[:10]:
    n = len(todos_motivos[motivo])
    print(f"  {motivo}: {score:.4f} (presente en {n}/25 secuencias)")

# ── FIGURA ───────────────────────────────────────────────────
nombres = [m[0] for m in top_motivos]
valores = [m[1] for m in top_motivos]
frecuencias = [len(todos_motivos[m[0]]) for m in top_motivos]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1 — Score promedio
axes[0].barh(nombres, valores, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("Mean IG Attribution Score (n=25 sequences)")
axes[0].set_title("Top 20 Universal KPC-3 Motifs\nMean IG score across 25 sequences", fontsize=11)

# Plot 2 — Frecuencia de aparición
axes[1].barh(nombres, frecuencias, color='darkred', alpha=0.85)
axes[1].invert_yaxis()
axes[1].axvline(x=25, color='black', linestyle='--', alpha=0.5, label='n=25')
axes[1].set_xlabel("Number of sequences containing motif (out of 25)")
axes[1].set_title("Motif Consistency\nHow many KPC-3 sequences share each motif", fontsize=11)
axes[1].legend()

plt.suptitle(
    "Figure — Population-Level Integrated Gradients Analysis\n"
    "Universal resistance motifs across 25 KPC-3 beta-lactamase sequences",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    '/content/drive/MyDrive/rgi_Klebsiella_pneumoniae_ertapenem/dataset_limpio/ig_kpc3_population.png',
    dpi=150, bbox_inches='tight'
)
plt.show()
print("✅ Guardado: ig_kpc3_population.png")
============================================================
=== Found in Flaming_AI(produccion) 2 con las cosas para responderle a este cuate.ipynb Cell 15 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        # CORRECCIÓN: Usar core.embed_layer en lugar de embeddings
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated = interpolated.detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    # ── MAPEAR ATRIBUCIONES A KMERS DE LA SECUENCIA ORIGINAL ─
    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)

    # Construir kmers con sus atribuciones
    kmer_scores = {}
    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        kmer_seq = seq_limpia[i:i+kmer]
        score = float(np.mean(token_importance[i:i+kmer]))
        if kmer_seq in kmer_scores:
            kmer_scores[kmer_seq] += score
        else:
            kmer_scores[kmer_seq] = score

    # Top 20 kmers por score total
    top_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\nTop 10 motivos 6-mer ({label_str}):")
    for km, sc in top_kmers[:10]:
        print(f"  {km}: {sc:.6f}")

    # ── FIGURA ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    nombres = [k[0] for k in top_kmers]
    valores = [k[1] for k in top_kmers]
    color = 'red' if label_str == 'Resistant' else 'blue'
    plt.barh(nombres, valores, color=color, alpha=0.85)
    plt.xlabel("Integrated Gradients Attribution (mean over k-mer positions)")
    plt.title(f"Figure — Integrated Gradients 6-mer Motifs: {label_str} sequence\n"
              f"(top 20 motifs by attribution score)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ruta = f'{RUTA_BASE}/ig_kmers_{label_str.lower()}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Guardado: ig_kmers_{label_str.lower()}.png")
    return top_kmers

# ── CORRER ───────────────────────────────────────────────────
idx_R = np.where(labels_muestra == 1)[0][0]
idx_S = np.where(labels_muestra == 0)[0][0]

print("🧬 Integrated Gradients 6-mer — Resistente...")
motivos_R = integrated_gradients_motifs(secuencias[idx_R], "Resistant", target_class=1)

print("\n🧬 Integrated Gradients 6-mer — Susceptible...")
motivos_S = integrated_gradients_motifs(secuencias[idx_S], "Susceptible", target_class=0)

============================================================
=== Found in Flaming_AI(produccion) 2 con las cosas para responderle a este cuate.ipynb Cell 16 ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Resistente
nombres_R = [k[0] for k in motivos_R[:15]]
valores_R = [k[1] for k in motivos_R[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("Resistant sequence\nTop 15 6-mer motifs", fontsize=12)

# Susceptible
nombres_S = [k[0] for k in motivos_S[:15]]
valores_S = [k[1] for k in motivos_S[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle("Figure X — Integrated Gradients 6-mer Motif Analysis\n"
             "NTv3 attribution scores for Ertapenem Resistance vs Susceptibility",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_IG_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X guardada")
============================================================
=== Found in Flaming_AI(produccion) 2 con las cosas para responderle a este cuate.ipynb Cell 22 ===
# IG sobre KPC-3 real
print("🧬 Integrated Gradients — KPC-3 (carbapenemasa clínica)...")
motivos_KPC3 = integrated_gradients_motifs(seq_kpc3, "KPC-3 Resistant", target_class=1)

# Buscar también una secuencia susceptible con gen bien identificado
# para comparación limpia
seq_sus = val_df[val_df['label'] == 0].iloc[0]['sequence']
print("\n🧬 Integrated Gradients — Susceptible...")
motivos_SUS = integrated_gradients_motifs(seq_sus, "Susceptible", target_class=0)

# Figura comparativa final
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

nombres_R = [k[0] for k in motivos_KPC3[:15]]
valores_R = [k[1] for k in motivos_KPC3[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("KPC-3 beta-lactamase (Resistant)\nTop 15 6-mer motifs", fontsize=12)

nombres_S = [k[0] for k in motivos_SUS[:15]]
valores_S = [k[1] for k in motivos_SUS[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle(
    "Figure X — Integrated Gradients 6-mer Motif Analysis\n"
    "KPC-3 beta-lactamase vs Susceptible: NTv3 attribution scores for Ertapenem resistance",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_KPC3_vs_susceptible.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X final guardada: figura_X_KPC3_vs_susceptible.png")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_completo).ipynb Cell 15 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        # CORRECCIÓN: Usar core.embed_layer en lugar de embeddings
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated = interpolated.detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    # ── MAPEAR ATRIBUCIONES A KMERS DE LA SECUENCIA ORIGINAL ─
    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)

    # Construir kmers con sus atribuciones
    kmer_scores = {}
    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        kmer_seq = seq_limpia[i:i+kmer]
        score = float(np.mean(token_importance[i:i+kmer]))
        if kmer_seq in kmer_scores:
            kmer_scores[kmer_seq] += score
        else:
            kmer_scores[kmer_seq] = score

    # Top 20 kmers por score total
    top_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\nTop 10 motivos 6-mer ({label_str}):")
    for km, sc in top_kmers[:10]:
        print(f"  {km}: {sc:.6f}")

    # ── FIGURA ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    nombres = [k[0] for k in top_kmers]
    valores = [k[1] for k in top_kmers]
    color = 'red' if label_str == 'Resistant' else 'blue'
    plt.barh(nombres, valores, color=color, alpha=0.85)
    plt.xlabel("Integrated Gradients Attribution (mean over k-mer positions)")
    plt.title(f"Figure — Integrated Gradients 6-mer Motifs: {label_str} sequence\n"
              f"(top 20 motifs by attribution score)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ruta = f'{RUTA_BASE}/ig_kmers_{label_str.lower()}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Guardado: ig_kmers_{label_str.lower()}.png")
    return top_kmers

# ── CORRER ───────────────────────────────────────────────────
idx_R = np.where(labels_muestra == 1)[0][0]
idx_S = np.where(labels_muestra == 0)[0][0]

print("🧬 Integrated Gradients 6-mer — Resistente...")
motivos_R = integrated_gradients_motifs(secuencias[idx_R], "Resistant", target_class=1)

print("\n🧬 Integrated Gradients 6-mer — Susceptible...")
motivos_S = integrated_gradients_motifs(secuencias[idx_S], "Susceptible", target_class=0)

============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_completo).ipynb Cell 16 ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Resistente
nombres_R = [k[0] for k in motivos_R[:15]]
valores_R = [k[1] for k in motivos_R[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("Resistant sequence\nTop 15 6-mer motifs", fontsize=12)

# Susceptible
nombres_S = [k[0] for k in motivos_S[:15]]
valores_S = [k[1] for k in motivos_S[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle("Figure X — Integrated Gradients 6-mer Motif Analysis\n"
             "NTv3 attribution scores for Ertapenem Resistance vs Susceptibility",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_IG_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X guardada")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_completo).ipynb Cell 22 ===
# IG sobre KPC-3 real
print("🧬 Integrated Gradients — KPC-3 (carbapenemasa clínica)...")
motivos_KPC3 = integrated_gradients_motifs(seq_kpc3, "KPC-3 Resistant", target_class=1)

# Buscar también una secuencia susceptible con gen bien identificado
# para comparación limpia
seq_sus = val_df[val_df['label'] == 0].iloc[0]['sequence']
print("\n🧬 Integrated Gradients — Susceptible...")
motivos_SUS = integrated_gradients_motifs(seq_sus, "Susceptible", target_class=0)

# Figura comparativa final
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

nombres_R = [k[0] for k in motivos_KPC3[:15]]
valores_R = [k[1] for k in motivos_KPC3[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("KPC-3 beta-lactamase (Resistant)\nTop 15 6-mer motifs", fontsize=12)

nombres_S = [k[0] for k in motivos_SUS[:15]]
valores_S = [k[1] for k in motivos_SUS[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle(
    "Figure X — Integrated Gradients 6-mer Motif Analysis\n"
    "KPC-3 beta-lactamase vs Susceptible: NTv3 attribution scores for Ertapenem resistance",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_KPC3_vs_susceptible.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X final guardada: figura_X_KPC3_vs_susceptible.png")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_completo).ipynb Cell 34 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask",
                    (input_ids != tokenizer.pad_token_id).long()).to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = (baseline_embeds + alpha *
                       (input_embeds - baseline_embeds)).detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads *
                   (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)
    kmer_scores = {}

    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        motivo = seq_limpia[i:i+kmer]
        if len(motivo) == kmer and all(c in 'ATGC' for c in motivo):
            score = float(np.mean(token_importance[i:i+kmer]))
            if motivo in kmer_scores:
                kmer_scores[motivo] = max(kmer_scores[motivo], score)
            else:
                kmer_scores[motivo] = score

    return sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("✅ Función integrated_gradients_motifs definida")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_completo).ipynb Cell 35 ===
# ── IG SOBRE LAS 25 KPC-3 ────────────────────────────────────
print("Corriendo IG sobre 25 secuencias KPC-3...")

todos_motivos = {}

for idx, row in tqdm(df_kpc3.iterrows(), total=len(df_kpc3), desc="IG KPC-3"):
    try:
        motivos = integrated_gradients_motifs(
            row['sequence'],
            label_str="KPC-3",
            target_class=1,
            steps=50,
            kmer=6
        )
        for motivo, score in motivos:
            if motivo not in todos_motivos:
                todos_motivos[motivo] = []
            todos_motivos[motivo].append(score)
    except Exception as e:
        print(f"Error en secuencia {idx}: {e}")
        continue

# ── AGREGAR POR MOTIVO ───────────────────────────────────────
motivos_agregados = {
    motivo: np.mean(scores)
    for motivo, scores in todos_motivos.items()
    if len(scores) >= 5  # aparece en al menos 5 de 25 secuencias
}

top_motivos = sorted(motivos_agregados.items(), key=lambda x: x[1], reverse=True)[:20]

print(f"\n✅ Motivos únicos encontrados: {len(todos_motivos)}")
print(f"Motivos consistentes (≥5 secuencias): {len(motivos_agregados)}")
print(f"\nTop 10 motivos universales KPC-3:")
for motivo, score in top_motivos[:10]:
    n = len(todos_motivos[motivo])
    print(f"  {motivo}: {score:.4f} (presente en {n}/25 secuencias)")

# ── FIGURA ───────────────────────────────────────────────────
nombres = [m[0] for m in top_motivos]
valores = [m[1] for m in top_motivos]
frecuencias = [len(todos_motivos[m[0]]) for m in top_motivos]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1 — Score promedio
axes[0].barh(nombres, valores, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("Mean IG Attribution Score (n=25 sequences)")
axes[0].set_title("Top 20 Universal KPC-3 Motifs\nMean IG score across 25 sequences", fontsize=11)

# Plot 2 — Frecuencia de aparición
axes[1].barh(nombres, frecuencias, color='darkred', alpha=0.85)
axes[1].invert_yaxis()
axes[1].axvline(x=25, color='black', linestyle='--', alpha=0.5, label='n=25')
axes[1].set_xlabel("Number of sequences containing motif (out of 25)")
axes[1].set_title("Motif Consistency\nHow many KPC-3 sequences share each motif", fontsize=11)
axes[1].legend()

plt.suptitle(
    "Figure — Population-Level Integrated Gradients Analysis\n"
    "Universal resistance motifs across 25 KPC-3 beta-lactamase sequences",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    '/content/drive/MyDrive/rgi_Klebsiella_pneumoniae_ertapenem/dataset_limpio/ig_kpc3_population.png',
    dpi=150, bbox_inches='tight'
)
plt.show()
print("✅ Guardado: ig_kpc3_population.png")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2) (1).ipynb Cell 16 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        # CORRECCIÓN: Usar core.embed_layer en lugar de embeddings
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated = interpolated.detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    # ── MAPEAR ATRIBUCIONES A KMERS DE LA SECUENCIA ORIGINAL ─
    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)

    # Construir kmers con sus atribuciones
    kmer_scores = {}
    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        kmer_seq = seq_limpia[i:i+kmer]
        score = float(np.mean(token_importance[i:i+kmer]))
        if kmer_seq in kmer_scores:
            kmer_scores[kmer_seq] += score
        else:
            kmer_scores[kmer_seq] = score

    # Top 20 kmers por score total
    top_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\nTop 10 motivos 6-mer ({label_str}):")
    for km, sc in top_kmers[:10]:
        print(f"  {km}: {sc:.6f}")

    # ── FIGURA ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    nombres = [k[0] for k in top_kmers]
    valores = [k[1] for k in top_kmers]
    color = 'red' if label_str == 'Resistant' else 'blue'
    plt.barh(nombres, valores, color=color, alpha=0.85)
    plt.xlabel("Integrated Gradients Attribution (mean over k-mer positions)")
    plt.title(f"Figure — Integrated Gradients 6-mer Motifs: {label_str} sequence\n"
              f"(top 20 motifs by attribution score)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ruta = f'{RUTA_BASE}/ig_kmers_{label_str.lower()}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Guardado: ig_kmers_{label_str.lower()}.png")
    return top_kmers

# ── CORRER ───────────────────────────────────────────────────
idx_R = np.where(labels_muestra == 1)[0][0]
idx_S = np.where(labels_muestra == 0)[0][0]

print("🧬 Integrated Gradients 6-mer — Resistente...")
motivos_R = integrated_gradients_motifs(secuencias[idx_R], "Resistant", target_class=1)

print("\n🧬 Integrated Gradients 6-mer — Susceptible...")
motivos_S = integrated_gradients_motifs(secuencias[idx_S], "Susceptible", target_class=0)

============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2) (1).ipynb Cell 17 ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Resistente
nombres_R = [k[0] for k in motivos_R[:15]]
valores_R = [k[1] for k in motivos_R[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("Resistant sequence\nTop 15 6-mer motifs", fontsize=12)

# Susceptible
nombres_S = [k[0] for k in motivos_S[:15]]
valores_S = [k[1] for k in motivos_S[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle("Figure X — Integrated Gradients 6-mer Motif Analysis\n"
             "NTv3 attribution scores for Ertapenem Resistance vs Susceptibility",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_IG_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X guardada")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2) (1).ipynb Cell 23 ===
# IG sobre KPC-3 real
print("🧬 Integrated Gradients — KPC-3 (carbapenemasa clínica)...")
motivos_KPC3 = integrated_gradients_motifs(seq_kpc3, "KPC-3 Resistant", target_class=1)

# Buscar también una secuencia susceptible con gen bien identificado
# para comparación limpia
seq_sus = val_df[val_df['label'] == 0].iloc[0]['sequence']
print("\n🧬 Integrated Gradients — Susceptible...")
motivos_SUS = integrated_gradients_motifs(seq_sus, "Susceptible", target_class=0)

# Figura comparativa final
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

nombres_R = [k[0] for k in motivos_KPC3[:15]]
valores_R = [k[1] for k in motivos_KPC3[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("KPC-3 beta-lactamase (Resistant)\nTop 15 6-mer motifs", fontsize=12)

nombres_S = [k[0] for k in motivos_SUS[:15]]
valores_S = [k[1] for k in motivos_SUS[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle(
    "Figure X — Integrated Gradients 6-mer Motif Analysis\n"
    "KPC-3 beta-lactamase vs Susceptible: NTv3 attribution scores for Ertapenem resistance",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_KPC3_vs_susceptible.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X final guardada: figura_X_KPC3_vs_susceptible.png")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2) (1).ipynb Cell 35 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask",
                    (input_ids != tokenizer.pad_token_id).long()).to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = (baseline_embeds + alpha *
                       (input_embeds - baseline_embeds)).detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads *
                   (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)
    kmer_scores = {}

    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        motivo = seq_limpia[i:i+kmer]
        if len(motivo) == kmer and all(c in 'ATGC' for c in motivo):
            score = float(np.mean(token_importance[i:i+kmer]))
            if motivo in kmer_scores:
                kmer_scores[motivo] = max(kmer_scores[motivo], score)
            else:
                kmer_scores[motivo] = score

    return sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("✅ Función integrated_gradients_motifs definida")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2) (1).ipynb Cell 36 ===
# ── IG SOBRE LAS 25 KPC-3 ────────────────────────────────────
print("Corriendo IG sobre 25 secuencias KPC-3...")

todos_motivos = {}

for idx, row in tqdm(df_kpc3.iterrows(), total=len(df_kpc3), desc="IG KPC-3"):
    try:
        motivos = integrated_gradients_motifs(
            row['sequence'],
            label_str="KPC-3",
            target_class=1,
            steps=50,
            kmer=6
        )
        for motivo, score in motivos:
            if motivo not in todos_motivos:
                todos_motivos[motivo] = []
            todos_motivos[motivo].append(score)
    except Exception as e:
        print(f"Error en secuencia {idx}: {e}")
        continue

# ── AGREGAR POR MOTIVO ───────────────────────────────────────
motivos_agregados = {
    motivo: np.mean(scores)
    for motivo, scores in todos_motivos.items()
    if len(scores) >= 5  # aparece en al menos 5 de 25 secuencias
}

top_motivos = sorted(motivos_agregados.items(), key=lambda x: x[1], reverse=True)[:20]

print(f"\n✅ Motivos únicos encontrados: {len(todos_motivos)}")
print(f"Motivos consistentes (≥5 secuencias): {len(motivos_agregados)}")
print(f"\nTop 10 motivos universales KPC-3:")
for motivo, score in top_motivos[:10]:
    n = len(todos_motivos[motivo])
    print(f"  {motivo}: {score:.4f} (presente en {n}/25 secuencias)")

# ── FIGURA ───────────────────────────────────────────────────
nombres = [m[0] for m in top_motivos]
valores = [m[1] for m in top_motivos]
frecuencias = [len(todos_motivos[m[0]]) for m in top_motivos]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1 — Score promedio
axes[0].barh(nombres, valores, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("Mean IG Attribution Score (n=25 sequences)")
axes[0].set_title("Top 20 Universal KPC-3 Motifs\nMean IG score across 25 sequences", fontsize=11)

# Plot 2 — Frecuencia de aparición
axes[1].barh(nombres, frecuencias, color='darkred', alpha=0.85)
axes[1].invert_yaxis()
axes[1].axvline(x=25, color='black', linestyle='--', alpha=0.5, label='n=25')
axes[1].set_xlabel("Number of sequences containing motif (out of 25)")
axes[1].set_title("Motif Consistency\nHow many KPC-3 sequences share each motif", fontsize=11)
axes[1].legend()

plt.suptitle(
    "Figure — Population-Level Integrated Gradients Analysis\n"
    "Universal resistance motifs across 25 KPC-3 beta-lactamase sequences",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    '/content/drive/MyDrive/rgi_Klebsiella_pneumoniae_ertapenem/dataset_limpio/ig_kpc3_population.png',
    dpi=150, bbox_inches='tight'
)
plt.show()
print("✅ Guardado: ig_kpc3_population.png")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2).ipynb Cell 14 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        # CORRECCIÓN: Usar core.embed_layer en lugar de embeddings
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated = interpolated.detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    # ── MAPEAR ATRIBUCIONES A KMERS DE LA SECUENCIA ORIGINAL ─
    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)

    # Construir kmers con sus atribuciones
    kmer_scores = {}
    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        kmer_seq = seq_limpia[i:i+kmer]
        score = float(np.mean(token_importance[i:i+kmer]))
        if kmer_seq in kmer_scores:
            kmer_scores[kmer_seq] += score
        else:
            kmer_scores[kmer_seq] = score

    # Top 20 kmers por score total
    top_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\nTop 10 motivos 6-mer ({label_str}):")
    for km, sc in top_kmers[:10]:
        print(f"  {km}: {sc:.6f}")

    # ── FIGURA ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    nombres = [k[0] for k in top_kmers]
    valores = [k[1] for k in top_kmers]
    color = 'red' if label_str == 'Resistant' else 'blue'
    plt.barh(nombres, valores, color=color, alpha=0.85)
    plt.xlabel("Integrated Gradients Attribution (mean over k-mer positions)")
    plt.title(f"Figure — Integrated Gradients 6-mer Motifs: {label_str} sequence\n"
              f"(top 20 motifs by attribution score)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ruta = f'{RUTA_BASE}/ig_kmers_{label_str.lower()}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Guardado: ig_kmers_{label_str.lower()}.png")
    return top_kmers

# ── CORRER ───────────────────────────────────────────────────
idx_R = np.where(labels_muestra == 1)[0][0]
idx_S = np.where(labels_muestra == 0)[0][0]

print("🧬 Integrated Gradients 6-mer — Resistente...")
motivos_R = integrated_gradients_motifs(secuencias[idx_R], "Resistant", target_class=1)

print("\n🧬 Integrated Gradients 6-mer — Susceptible...")
motivos_S = integrated_gradients_motifs(secuencias[idx_S], "Susceptible", target_class=0)

============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2).ipynb Cell 15 ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Resistente
nombres_R = [k[0] for k in motivos_R[:15]]
valores_R = [k[1] for k in motivos_R[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("Resistant sequence\nTop 15 6-mer motifs", fontsize=12)

# Susceptible
nombres_S = [k[0] for k in motivos_S[:15]]
valores_S = [k[1] for k in motivos_S[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle("Figure X — Integrated Gradients 6-mer Motif Analysis\n"
             "NTv3 attribution scores for Ertapenem Resistance vs Susceptibility",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_IG_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X guardada")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2).ipynb Cell 21 ===
# IG sobre KPC-3 real
print("🧬 Integrated Gradients — KPC-3 (carbapenemasa clínica)...")
motivos_KPC3 = integrated_gradients_motifs(seq_kpc3, "KPC-3 Resistant", target_class=1)

# Buscar también una secuencia susceptible con gen bien identificado
# para comparación limpia
seq_sus = val_df[val_df['label'] == 0].iloc[0]['sequence']
print("\n🧬 Integrated Gradients — Susceptible...")
motivos_SUS = integrated_gradients_motifs(seq_sus, "Susceptible", target_class=0)

# Figura comparativa final
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

nombres_R = [k[0] for k in motivos_KPC3[:15]]
valores_R = [k[1] for k in motivos_KPC3[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("KPC-3 beta-lactamase (Resistant)\nTop 15 6-mer motifs", fontsize=12)

nombres_S = [k[0] for k in motivos_SUS[:15]]
valores_S = [k[1] for k in motivos_SUS[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle(
    "Figure X — Integrated Gradients 6-mer Motif Analysis\n"
    "KPC-3 beta-lactamase vs Susceptible: NTv3 attribution scores for Ertapenem resistance",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_KPC3_vs_susceptible.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X final guardada: figura_X_KPC3_vs_susceptible.png")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2).ipynb Cell 33 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask",
                    (input_ids != tokenizer.pad_token_id).long()).to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = (baseline_embeds + alpha *
                       (input_embeds - baseline_embeds)).detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads *
                   (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)
    kmer_scores = {}

    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        motivo = seq_limpia[i:i+kmer]
        if len(motivo) == kmer and all(c in 'ATGC' for c in motivo):
            score = float(np.mean(token_importance[i:i+kmer]))
            if motivo in kmer_scores:
                kmer_scores[motivo] = max(kmer_scores[motivo], score)
            else:
                kmer_scores[motivo] = score

    return sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("✅ Función integrated_gradients_motifs definida")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2).ipynb Cell 34 ===
# ── IG SOBRE LAS 25 KPC-3 ────────────────────────────────────
print("Corriendo IG sobre 25 secuencias KPC-3...")

todos_motivos = {}

for idx, row in tqdm(df_kpc3.iterrows(), total=len(df_kpc3), desc="IG KPC-3"):
    try:
        motivos = integrated_gradients_motifs(
            row['sequence'],
            label_str="KPC-3",
            target_class=1,
            steps=50,
            kmer=6
        )
        for motivo, score in motivos:
            if motivo not in todos_motivos:
                todos_motivos[motivo] = []
            todos_motivos[motivo].append(score)
    except Exception as e:
        print(f"Error en secuencia {idx}: {e}")
        continue

# ── AGREGAR POR MOTIVO ───────────────────────────────────────
motivos_agregados = {
    motivo: np.mean(scores)
    for motivo, scores in todos_motivos.items()
    if len(scores) >= 5  # aparece en al menos 5 de 25 secuencias
}

top_motivos = sorted(motivos_agregados.items(), key=lambda x: x[1], reverse=True)[:20]

print(f"\n✅ Motivos únicos encontrados: {len(todos_motivos)}")
print(f"Motivos consistentes (≥5 secuencias): {len(motivos_agregados)}")
print(f"\nTop 10 motivos universales KPC-3:")
for motivo, score in top_motivos[:10]:
    n = len(todos_motivos[motivo])
    print(f"  {motivo}: {score:.4f} (presente en {n}/25 secuencias)")

# ── FIGURA ───────────────────────────────────────────────────
nombres = [m[0] for m in top_motivos]
valores = [m[1] for m in top_motivos]
frecuencias = [len(todos_motivos[m[0]]) for m in top_motivos]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1 — Score promedio
axes[0].barh(nombres, valores, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("Mean IG Attribution Score (n=25 sequences)")
axes[0].set_title("Top 20 Universal KPC-3 Motifs\nMean IG score across 25 sequences", fontsize=11)

# Plot 2 — Frecuencia de aparición
axes[1].barh(nombres, frecuencias, color='darkred', alpha=0.85)
axes[1].invert_yaxis()
axes[1].axvline(x=25, color='black', linestyle='--', alpha=0.5, label='n=25')
axes[1].set_xlabel("Number of sequences containing motif (out of 25)")
axes[1].set_title("Motif Consistency\nHow many KPC-3 sequences share each motif", fontsize=11)
axes[1].legend()

plt.suptitle(
    "Figure — Population-Level Integrated Gradients Analysis\n"
    "Universal resistance motifs across 25 KPC-3 beta-lactamase sequences",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    '/content/drive/MyDrive/rgi_Klebsiella_pneumoniae_ertapenem/dataset_limpio/ig_kpc3_population.png',
    dpi=150, bbox_inches='tight'
)
plt.show()
print("✅ Guardado: ig_kpc3_population.png")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2)_(1).ipynb Cell 16 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        # CORRECCIÓN: Usar core.embed_layer en lugar de embeddings
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated = interpolated.detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    # ── MAPEAR ATRIBUCIONES A KMERS DE LA SECUENCIA ORIGINAL ─
    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)

    # Construir kmers con sus atribuciones
    kmer_scores = {}
    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        kmer_seq = seq_limpia[i:i+kmer]
        score = float(np.mean(token_importance[i:i+kmer]))
        if kmer_seq in kmer_scores:
            kmer_scores[kmer_seq] += score
        else:
            kmer_scores[kmer_seq] = score

    # Top 20 kmers por score total
    top_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\nTop 10 motivos 6-mer ({label_str}):")
    for km, sc in top_kmers[:10]:
        print(f"  {km}: {sc:.6f}")

    # ── FIGURA ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    nombres = [k[0] for k in top_kmers]
    valores = [k[1] for k in top_kmers]
    color = 'red' if label_str == 'Resistant' else 'blue'
    plt.barh(nombres, valores, color=color, alpha=0.85)
    plt.xlabel("Integrated Gradients Attribution (mean over k-mer positions)")
    plt.title(f"Figure — Integrated Gradients 6-mer Motifs: {label_str} sequence\n"
              f"(top 20 motifs by attribution score)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ruta = f'{RUTA_BASE}/ig_kmers_{label_str.lower()}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Guardado: ig_kmers_{label_str.lower()}.png")
    return top_kmers

# ── CORRER ───────────────────────────────────────────────────
idx_R = np.where(labels_muestra == 1)[0][0]
idx_S = np.where(labels_muestra == 0)[0][0]

print("🧬 Integrated Gradients 6-mer — Resistente...")
motivos_R = integrated_gradients_motifs(secuencias[idx_R], "Resistant", target_class=1)

print("\n🧬 Integrated Gradients 6-mer — Susceptible...")
motivos_S = integrated_gradients_motifs(secuencias[idx_S], "Susceptible", target_class=0)

============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2)_(1).ipynb Cell 17 ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Resistente
nombres_R = [k[0] for k in motivos_R[:15]]
valores_R = [k[1] for k in motivos_R[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("Resistant sequence\nTop 15 6-mer motifs", fontsize=12)

# Susceptible
nombres_S = [k[0] for k in motivos_S[:15]]
valores_S = [k[1] for k in motivos_S[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle("Figure X — Integrated Gradients 6-mer Motif Analysis\n"
             "NTv3 attribution scores for Ertapenem Resistance vs Susceptibility",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_IG_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X guardada")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2)_(1).ipynb Cell 23 ===
# IG sobre KPC-3 real
print("🧬 Integrated Gradients — KPC-3 (carbapenemasa clínica)...")
motivos_KPC3 = integrated_gradients_motifs(seq_kpc3, "KPC-3 Resistant", target_class=1)

# Buscar también una secuencia susceptible con gen bien identificado
# para comparación limpia
seq_sus = val_df[val_df['label'] == 0].iloc[0]['sequence']
print("\n🧬 Integrated Gradients — Susceptible...")
motivos_SUS = integrated_gradients_motifs(seq_sus, "Susceptible", target_class=0)

# Figura comparativa final
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

nombres_R = [k[0] for k in motivos_KPC3[:15]]
valores_R = [k[1] for k in motivos_KPC3[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("KPC-3 beta-lactamase (Resistant)\nTop 15 6-mer motifs", fontsize=12)

nombres_S = [k[0] for k in motivos_SUS[:15]]
valores_S = [k[1] for k in motivos_SUS[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle(
    "Figure X — Integrated Gradients 6-mer Motif Analysis\n"
    "KPC-3 beta-lactamase vs Susceptible: NTv3 attribution scores for Ertapenem resistance",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_KPC3_vs_susceptible.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X final guardada: figura_X_KPC3_vs_susceptible.png")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2)_(1).ipynb Cell 35 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask",
                    (input_ids != tokenizer.pad_token_id).long()).to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = (baseline_embeds + alpha *
                       (input_embeds - baseline_embeds)).detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads *
                   (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)
    kmer_scores = {}

    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        motivo = seq_limpia[i:i+kmer]
        if len(motivo) == kmer and all(c in 'ATGC' for c in motivo):
            score = float(np.mean(token_importance[i:i+kmer]))
            if motivo in kmer_scores:
                kmer_scores[motivo] = max(kmer_scores[motivo], score)
            else:
                kmer_scores[motivo] = score

    return sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("✅ Función integrated_gradients_motifs definida")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v2)_(1).ipynb Cell 36 ===
# ── IG SOBRE LAS 25 KPC-3 ────────────────────────────────────
print("Corriendo IG sobre 25 secuencias KPC-3...")

todos_motivos = {}

for idx, row in tqdm(df_kpc3.iterrows(), total=len(df_kpc3), desc="IG KPC-3"):
    try:
        motivos = integrated_gradients_motifs(
            row['sequence'],
            label_str="KPC-3",
            target_class=1,
            steps=50,
            kmer=6
        )
        for motivo, score in motivos:
            if motivo not in todos_motivos:
                todos_motivos[motivo] = []
            todos_motivos[motivo].append(score)
    except Exception as e:
        print(f"Error en secuencia {idx}: {e}")
        continue

# ── AGREGAR POR MOTIVO ───────────────────────────────────────
motivos_agregados = {
    motivo: np.mean(scores)
    for motivo, scores in todos_motivos.items()
    if len(scores) >= 5  # aparece en al menos 5 de 25 secuencias
}

top_motivos = sorted(motivos_agregados.items(), key=lambda x: x[1], reverse=True)[:20]

print(f"\n✅ Motivos únicos encontrados: {len(todos_motivos)}")
print(f"Motivos consistentes (≥5 secuencias): {len(motivos_agregados)}")
print(f"\nTop 10 motivos universales KPC-3:")
for motivo, score in top_motivos[:10]:
    n = len(todos_motivos[motivo])
    print(f"  {motivo}: {score:.4f} (presente en {n}/25 secuencias)")

# ── FIGURA ───────────────────────────────────────────────────
nombres = [m[0] for m in top_motivos]
valores = [m[1] for m in top_motivos]
frecuencias = [len(todos_motivos[m[0]]) for m in top_motivos]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1 — Score promedio
axes[0].barh(nombres, valores, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("Mean IG Attribution Score (n=25 sequences)")
axes[0].set_title("Top 20 Universal KPC-3 Motifs\nMean IG score across 25 sequences", fontsize=11)

# Plot 2 — Frecuencia de aparición
axes[1].barh(nombres, frecuencias, color='darkred', alpha=0.85)
axes[1].invert_yaxis()
axes[1].axvline(x=25, color='black', linestyle='--', alpha=0.5, label='n=25')
axes[1].set_xlabel("Number of sequences containing motif (out of 25)")
axes[1].set_title("Motif Consistency\nHow many KPC-3 sequences share each motif", fontsize=11)
axes[1].legend()

plt.suptitle(
    "Figure — Population-Level Integrated Gradients Analysis\n"
    "Universal resistance motifs across 25 KPC-3 beta-lactamase sequences",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    '/content/drive/MyDrive/rgi_Klebsiella_pneumoniae_ertapenem/dataset_limpio/ig_kpc3_population.png',
    dpi=150, bbox_inches='tight'
)
plt.show()
print("✅ Guardado: ig_kpc3_population.png")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (1).ipynb Cell 18 ===
import torch
import numpy as np
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

def analizar_interpretabilidad_mil(model, tokenizer, bag, device="cuda"):
    model.to(device)
    model.eval()

    sequences = bag["sequences"]
    genome_id = bag.get("genome_id", "Unknown")
    label = bag.get("label", "?")

    print(f"🔍 Analizando Genoma: {genome_id} (Label: {label})")

    # --- PASO 1: Identificar el Gen con mayor Probabilidad (Top-1) ---
    all_probs = []
    sub_batch = 8

    with torch.no_grad():
        for start in range(0, len(sequences), sub_batch):
            chunk = sequences[start : start + sub_batch]
            inputs = tokenizer(chunk, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
            sp_ids = torch.full((inputs['input_ids'].size(0),), 6, device=device)

            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'), species_ids=sp_ids)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs)

    all_probs = torch.cat(all_probs)
    top_instance_idx = torch.argmax(all_probs).item()
    top_sequence = sequences[top_instance_idx]

    print(f"📍 Gen detectado como 'Driver' de Resistencia: {top_sequence[:60]}...")
    print(f"🔥 Probabilidad del Gen: {all_probs[top_instance_idx].item():.4f}")

    # --- PASO 2: Integrated Gradients sobre el Gen Ganador ---
    def predict_wrapper(inputs_embeds):
        # FIX: Expand species_ids to match the batch size created by Captum steps
        batch_size = inputs_embeds.shape[0]
        sp_id = torch.full((batch_size,), 6, device=device)

        outputs = model.ntv3(inputs_embeds=inputs_embeds, species_ids=sp_id, output_hidden_states=True)
        last_hidden_layer = outputs.hidden_states[-1]
        cls_output = last_hidden_layer[:, 0, :]
        logits = model.classifier(model.dropout(cls_output))
        return logits[:, 1]

    ig = IntegratedGradients(predict_wrapper)

    inputs_top = tokenizer(top_sequence, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs_top.input_ids

    with torch.no_grad():
        input_embeds = model.ntv3.core.embed_layer(input_ids)

    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    baseline_embeds = model.ntv3.core.embed_layer(baseline_ids)

    # Run attribution
    attributions = ig.attribute(inputs=input_embeds, baselines=baseline_embeds, target=None)

    scores = attributions.squeeze(0).norm(p=2, dim=-1).cpu().detach().numpy()

    # --- PASO 3: Visualización ---
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(scores)), scores, color='crimson', alpha=0.7)
    plt.title(f"Importancia (Integrated Gradients) - Gen Driver en {genome_id}")
    plt.xlabel("Posición del Token / Nucleótido")
    plt.ylabel("Atribución")
    plt.grid(alpha=0.2)
    plt.show()

    return top_sequence, scores, tokenizer.convert_ids_to_tokens(input_ids[0])
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (1).ipynb Cell 28 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        # CORRECCIÓN: Usar core.embed_layer en lugar de embeddings
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated = interpolated.detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    # ── MAPEAR ATRIBUCIONES A KMERS DE LA SECUENCIA ORIGINAL ─
    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)

    # Construir kmers con sus atribuciones
    kmer_scores = {}
    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        kmer_seq = seq_limpia[i:i+kmer]
        score = float(np.mean(token_importance[i:i+kmer]))
        if kmer_seq in kmer_scores:
            kmer_scores[kmer_seq] += score
        else:
            kmer_scores[kmer_seq] = score

    # Top 20 kmers por score total
    top_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\nTop 10 motivos 6-mer ({label_str}):")
    for km, sc in top_kmers[:10]:
        print(f"  {km}: {sc:.6f}")

    # ── FIGURA ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    nombres = [k[0] for k in top_kmers]
    valores = [k[1] for k in top_kmers]
    color = 'red' if label_str == 'Resistant' else 'blue'
    plt.barh(nombres, valores, color=color, alpha=0.85)
    plt.xlabel("Integrated Gradients Attribution (mean over k-mer positions)")
    plt.title(f"Figure — Integrated Gradients 6-mer Motifs: {label_str} sequence\n"
              f"(top 20 motifs by attribution score)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ruta = f'{RUTA_BASE}/ig_kmers_{label_str.lower()}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Guardado: ig_kmers_{label_str.lower()}.png")
    return top_kmers

# ── CORRER ───────────────────────────────────────────────────
idx_R = np.where(labels_muestra == 1)[0][0]
idx_S = np.where(labels_muestra == 0)[0][0]

print("🧬 Integrated Gradients 6-mer — Resistente...")
motivos_R = integrated_gradients_motifs(secuencias[idx_R], "Resistant", target_class=1)

print("\n🧬 Integrated Gradients 6-mer — Susceptible...")
motivos_S = integrated_gradients_motifs(secuencias[idx_S], "Susceptible", target_class=0)

============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (1).ipynb Cell 29 ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Resistente
nombres_R = [k[0] for k in motivos_R[:15]]
valores_R = [k[1] for k in motivos_R[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("Resistant sequence\nTop 15 6-mer motifs", fontsize=12)

# Susceptible
nombres_S = [k[0] for k in motivos_S[:15]]
valores_S = [k[1] for k in motivos_S[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle("Figure X — Integrated Gradients 6-mer Motif Analysis\n"
             "NTv3 attribution scores for Ertapenem Resistance vs Susceptibility",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_IG_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X guardada")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (1).ipynb Cell 40 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask",
                    (input_ids != tokenizer.pad_token_id).long()).to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = (baseline_embeds + alpha *
                       (input_embeds - baseline_embeds)).detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads *
                   (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)
    kmer_scores = {}

    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        motivo = seq_limpia[i:i+kmer]
        if len(motivo) == kmer and all(c in 'ATGC' for c in motivo):
            score = float(np.mean(token_importance[i:i+kmer]))
            if motivo in kmer_scores:
                kmer_scores[motivo] = max(kmer_scores[motivo], score)
            else:
                kmer_scores[motivo] = score

    return sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("✅ Función integrated_gradients_motifs definida")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (2).ipynb Cell 18 ===
import torch
import numpy as np
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

def analizar_interpretabilidad_mil(model, tokenizer, bag, device="cuda"):
    model.to(device)
    model.eval()

    sequences = bag["sequences"]
    genome_id = bag.get("genome_id", "Unknown")
    label = bag.get("label", "?")

    print(f"🔍 Analizando Genoma: {genome_id} (Label: {label})")

    # --- PASO 1: Identificar el Gen con mayor Probabilidad (Top-1) ---
    all_probs = []
    sub_batch = 8

    with torch.no_grad():
        for start in range(0, len(sequences), sub_batch):
            chunk = sequences[start : start + sub_batch]
            inputs = tokenizer(chunk, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
            sp_ids = torch.full((inputs['input_ids'].size(0),), 6, device=device)

            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'), species_ids=sp_ids)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs)

    all_probs = torch.cat(all_probs)
    top_instance_idx = torch.argmax(all_probs).item()
    top_sequence = sequences[top_instance_idx]

    print(f"📍 Gen detectado como 'Driver' de Resistencia: {top_sequence[:60]}...")
    print(f"🔥 Probabilidad del Gen: {all_probs[top_instance_idx].item():.4f}")

    # --- PASO 2: Integrated Gradients sobre el Gen Ganador ---
    def predict_wrapper(inputs_embeds):
        # FIX: Expand species_ids to match the batch size created by Captum steps
        batch_size = inputs_embeds.shape[0]
        sp_id = torch.full((batch_size,), 6, device=device)

        outputs = model.ntv3(inputs_embeds=inputs_embeds, species_ids=sp_id, output_hidden_states=True)
        last_hidden_layer = outputs.hidden_states[-1]
        cls_output = last_hidden_layer[:, 0, :]
        logits = model.classifier(model.dropout(cls_output))
        return logits[:, 1]

    ig = IntegratedGradients(predict_wrapper)

    inputs_top = tokenizer(top_sequence, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs_top.input_ids

    with torch.no_grad():
        input_embeds = model.ntv3.core.embed_layer(input_ids)

    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    baseline_embeds = model.ntv3.core.embed_layer(baseline_ids)

    # Run attribution
    attributions = ig.attribute(inputs=input_embeds, baselines=baseline_embeds, target=None)

    scores = attributions.squeeze(0).norm(p=2, dim=-1).cpu().detach().numpy()

    # --- PASO 3: Visualización ---
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(scores)), scores, color='crimson', alpha=0.7)
    plt.title(f"Importancia (Integrated Gradients) - Gen Driver en {genome_id}")
    plt.xlabel("Posición del Token / Nucleótido")
    plt.ylabel("Atribución")
    plt.grid(alpha=0.2)
    plt.show()

    return top_sequence, scores, tokenizer.convert_ids_to_tokens(input_ids[0])
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (2).ipynb Cell 28 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        # CORRECCIÓN: Usar core.embed_layer en lugar de embeddings
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated = interpolated.detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    # ── MAPEAR ATRIBUCIONES A KMERS DE LA SECUENCIA ORIGINAL ─
    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)

    # Construir kmers con sus atribuciones
    kmer_scores = {}
    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        kmer_seq = seq_limpia[i:i+kmer]
        score = float(np.mean(token_importance[i:i+kmer]))
        if kmer_seq in kmer_scores:
            kmer_scores[kmer_seq] += score
        else:
            kmer_scores[kmer_seq] = score

    # Top 20 kmers por score total
    top_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\nTop 10 motivos 6-mer ({label_str}):")
    for km, sc in top_kmers[:10]:
        print(f"  {km}: {sc:.6f}")

    # ── FIGURA ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    nombres = [k[0] for k in top_kmers]
    valores = [k[1] for k in top_kmers]
    color = 'red' if label_str == 'Resistant' else 'blue'
    plt.barh(nombres, valores, color=color, alpha=0.85)
    plt.xlabel("Integrated Gradients Attribution (mean over k-mer positions)")
    plt.title(f"Figure — Integrated Gradients 6-mer Motifs: {label_str} sequence\n"
              f"(top 20 motifs by attribution score)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ruta = f'{RUTA_BASE}/ig_kmers_{label_str.lower()}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Guardado: ig_kmers_{label_str.lower()}.png")
    return top_kmers

# ── CORRER ───────────────────────────────────────────────────
idx_R = np.where(labels_muestra == 1)[0][0]
idx_S = np.where(labels_muestra == 0)[0][0]

print("🧬 Integrated Gradients 6-mer — Resistente...")
motivos_R = integrated_gradients_motifs(secuencias[idx_R], "Resistant", target_class=1)

print("\n🧬 Integrated Gradients 6-mer — Susceptible...")
motivos_S = integrated_gradients_motifs(secuencias[idx_S], "Susceptible", target_class=0)

============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (2).ipynb Cell 29 ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Resistente
nombres_R = [k[0] for k in motivos_R[:15]]
valores_R = [k[1] for k in motivos_R[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("Resistant sequence\nTop 15 6-mer motifs", fontsize=12)

# Susceptible
nombres_S = [k[0] for k in motivos_S[:15]]
valores_S = [k[1] for k in motivos_S[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle("Figure X — Integrated Gradients 6-mer Motif Analysis\n"
             "NTv3 attribution scores for Ertapenem Resistance vs Susceptibility",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_IG_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X guardada")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (2).ipynb Cell 40 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask",
                    (input_ids != tokenizer.pad_token_id).long()).to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = (baseline_embeds + alpha *
                       (input_embeds - baseline_embeds)).detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads *
                   (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)
    kmer_scores = {}

    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        motivo = seq_limpia[i:i+kmer]
        if len(motivo) == kmer and all(c in 'ATGC' for c in motivo):
            score = float(np.mean(token_importance[i:i+kmer]))
            if motivo in kmer_scores:
                kmer_scores[motivo] = max(kmer_scores[motivo], score)
            else:
                kmer_scores[motivo] = score

    return sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("✅ Función integrated_gradients_motifs definida")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (3).ipynb Cell 18 ===
import torch
import numpy as np
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

def analizar_interpretabilidad_mil(model, tokenizer, bag, device="cuda"):
    model.to(device)
    model.eval()

    sequences = bag["sequences"]
    genome_id = bag.get("genome_id", "Unknown")
    label = bag.get("label", "?")

    print(f"🔍 Analizando Genoma: {genome_id} (Label: {label})")

    # --- PASO 1: Identificar el Gen con mayor Probabilidad (Top-1) ---
    all_probs = []
    sub_batch = 8

    with torch.no_grad():
        for start in range(0, len(sequences), sub_batch):
            chunk = sequences[start : start + sub_batch]
            inputs = tokenizer(chunk, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
            sp_ids = torch.full((inputs['input_ids'].size(0),), 6, device=device)

            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'), species_ids=sp_ids)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs)

    all_probs = torch.cat(all_probs)
    top_instance_idx = torch.argmax(all_probs).item()
    top_sequence = sequences[top_instance_idx]

    print(f"📍 Gen detectado como 'Driver' de Resistencia: {top_sequence[:60]}...")
    print(f"🔥 Probabilidad del Gen: {all_probs[top_instance_idx].item():.4f}")

    # --- PASO 2: Integrated Gradients sobre el Gen Ganador ---
    def predict_wrapper(inputs_embeds):
        # FIX: Expand species_ids to match the batch size created by Captum steps
        batch_size = inputs_embeds.shape[0]
        sp_id = torch.full((batch_size,), 6, device=device)

        outputs = model.ntv3(inputs_embeds=inputs_embeds, species_ids=sp_id, output_hidden_states=True)
        last_hidden_layer = outputs.hidden_states[-1]
        cls_output = last_hidden_layer[:, 0, :]
        logits = model.classifier(model.dropout(cls_output))
        return logits[:, 1]

    ig = IntegratedGradients(predict_wrapper)

    inputs_top = tokenizer(top_sequence, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs_top.input_ids

    with torch.no_grad():
        input_embeds = model.ntv3.core.embed_layer(input_ids)

    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    baseline_embeds = model.ntv3.core.embed_layer(baseline_ids)

    # Run attribution
    attributions = ig.attribute(inputs=input_embeds, baselines=baseline_embeds, target=None)

    scores = attributions.squeeze(0).norm(p=2, dim=-1).cpu().detach().numpy()

    # --- PASO 3: Visualización ---
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(scores)), scores, color='crimson', alpha=0.7)
    plt.title(f"Importancia (Integrated Gradients) - Gen Driver en {genome_id}")
    plt.xlabel("Posición del Token / Nucleótido")
    plt.ylabel("Atribución")
    plt.grid(alpha=0.2)
    plt.show()

    return top_sequence, scores, tokenizer.convert_ids_to_tokens(input_ids[0])
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (3).ipynb Cell 28 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        # CORRECCIÓN: Usar core.embed_layer en lugar de embeddings
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated = interpolated.detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    # ── MAPEAR ATRIBUCIONES A KMERS DE LA SECUENCIA ORIGINAL ─
    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)

    # Construir kmers con sus atribuciones
    kmer_scores = {}
    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        kmer_seq = seq_limpia[i:i+kmer]
        score = float(np.mean(token_importance[i:i+kmer]))
        if kmer_seq in kmer_scores:
            kmer_scores[kmer_seq] += score
        else:
            kmer_scores[kmer_seq] = score

    # Top 20 kmers por score total
    top_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\nTop 10 motivos 6-mer ({label_str}):")
    for km, sc in top_kmers[:10]:
        print(f"  {km}: {sc:.6f}")

    # ── FIGURA ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    nombres = [k[0] for k in top_kmers]
    valores = [k[1] for k in top_kmers]
    color = 'red' if label_str == 'Resistant' else 'blue'
    plt.barh(nombres, valores, color=color, alpha=0.85)
    plt.xlabel("Integrated Gradients Attribution (mean over k-mer positions)")
    plt.title(f"Figure — Integrated Gradients 6-mer Motifs: {label_str} sequence\n"
              f"(top 20 motifs by attribution score)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ruta = f'{RUTA_BASE}/ig_kmers_{label_str.lower()}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Guardado: ig_kmers_{label_str.lower()}.png")
    return top_kmers

# ── CORRER ───────────────────────────────────────────────────
idx_R = np.where(labels_muestra == 1)[0][0]
idx_S = np.where(labels_muestra == 0)[0][0]

print("🧬 Integrated Gradients 6-mer — Resistente...")
motivos_R = integrated_gradients_motifs(secuencias[idx_R], "Resistant", target_class=1)

print("\n🧬 Integrated Gradients 6-mer — Susceptible...")
motivos_S = integrated_gradients_motifs(secuencias[idx_S], "Susceptible", target_class=0)

============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (3).ipynb Cell 29 ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Resistente
nombres_R = [k[0] for k in motivos_R[:15]]
valores_R = [k[1] for k in motivos_R[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("Resistant sequence\nTop 15 6-mer motifs", fontsize=12)

# Susceptible
nombres_S = [k[0] for k in motivos_S[:15]]
valores_S = [k[1] for k in motivos_S[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle("Figure X — Integrated Gradients 6-mer Motif Analysis\n"
             "NTv3 attribution scores for Ertapenem Resistance vs Susceptibility",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_IG_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X guardada")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (3).ipynb Cell 40 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask",
                    (input_ids != tokenizer.pad_token_id).long()).to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = (baseline_embeds + alpha *
                       (input_embeds - baseline_embeds)).detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads *
                   (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)
    kmer_scores = {}

    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        motivo = seq_limpia[i:i+kmer]
        if len(motivo) == kmer and all(c in 'ATGC' for c in motivo):
            score = float(np.mean(token_importance[i:i+kmer]))
            if motivo in kmer_scores:
                kmer_scores[motivo] = max(kmer_scores[motivo], score)
            else:
                kmer_scores[motivo] = score

    return sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("✅ Función integrated_gradients_motifs definida")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (4).ipynb Cell 18 ===
import torch
import numpy as np
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

def analizar_interpretabilidad_mil(model, tokenizer, bag, device="cuda"):
    model.to(device)
    model.eval()

    sequences = bag["sequences"]
    genome_id = bag.get("genome_id", "Unknown")
    label = bag.get("label", "?")

    print(f"🔍 Analizando Genoma: {genome_id} (Label: {label})")

    # --- PASO 1: Identificar el Gen con mayor Probabilidad (Top-1) ---
    all_probs = []
    sub_batch = 8

    with torch.no_grad():
        for start in range(0, len(sequences), sub_batch):
            chunk = sequences[start : start + sub_batch]
            inputs = tokenizer(chunk, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
            sp_ids = torch.full((inputs['input_ids'].size(0),), 6, device=device)

            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'), species_ids=sp_ids)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs)

    all_probs = torch.cat(all_probs)
    top_instance_idx = torch.argmax(all_probs).item()
    top_sequence = sequences[top_instance_idx]

    print(f"📍 Gen detectado como 'Driver' de Resistencia: {top_sequence[:60]}...")
    print(f"🔥 Probabilidad del Gen: {all_probs[top_instance_idx].item():.4f}")

    # --- PASO 2: Integrated Gradients sobre el Gen Ganador ---
    def predict_wrapper(inputs_embeds):
        # FIX: Expand species_ids to match the batch size created by Captum steps
        batch_size = inputs_embeds.shape[0]
        sp_id = torch.full((batch_size,), 6, device=device)

        outputs = model.ntv3(inputs_embeds=inputs_embeds, species_ids=sp_id, output_hidden_states=True)
        last_hidden_layer = outputs.hidden_states[-1]
        cls_output = last_hidden_layer[:, 0, :]
        logits = model.classifier(model.dropout(cls_output))
        return logits[:, 1]

    ig = IntegratedGradients(predict_wrapper)

    inputs_top = tokenizer(top_sequence, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs_top.input_ids

    with torch.no_grad():
        input_embeds = model.ntv3.core.embed_layer(input_ids)

    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    baseline_embeds = model.ntv3.core.embed_layer(baseline_ids)

    # Run attribution
    attributions = ig.attribute(inputs=input_embeds, baselines=baseline_embeds, target=None)

    scores = attributions.squeeze(0).norm(p=2, dim=-1).cpu().detach().numpy()

    # --- PASO 3: Visualización ---
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(scores)), scores, color='crimson', alpha=0.7)
    plt.title(f"Importancia (Integrated Gradients) - Gen Driver en {genome_id}")
    plt.xlabel("Posición del Token / Nucleótido")
    plt.ylabel("Atribución")
    plt.grid(alpha=0.2)
    plt.show()

    return top_sequence, scores, tokenizer.convert_ids_to_tokens(input_ids[0])
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (4).ipynb Cell 28 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        # CORRECCIÓN: Usar core.embed_layer en lugar de embeddings
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated = interpolated.detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    # ── MAPEAR ATRIBUCIONES A KMERS DE LA SECUENCIA ORIGINAL ─
    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)

    # Construir kmers con sus atribuciones
    kmer_scores = {}
    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        kmer_seq = seq_limpia[i:i+kmer]
        score = float(np.mean(token_importance[i:i+kmer]))
        if kmer_seq in kmer_scores:
            kmer_scores[kmer_seq] += score
        else:
            kmer_scores[kmer_seq] = score

    # Top 20 kmers por score total
    top_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\nTop 10 motivos 6-mer ({label_str}):")
    for km, sc in top_kmers[:10]:
        print(f"  {km}: {sc:.6f}")

    # ── FIGURA ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    nombres = [k[0] for k in top_kmers]
    valores = [k[1] for k in top_kmers]
    color = 'red' if label_str == 'Resistant' else 'blue'
    plt.barh(nombres, valores, color=color, alpha=0.85)
    plt.xlabel("Integrated Gradients Attribution (mean over k-mer positions)")
    plt.title(f"Figure — Integrated Gradients 6-mer Motifs: {label_str} sequence\n"
              f"(top 20 motifs by attribution score)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ruta = f'{RUTA_BASE}/ig_kmers_{label_str.lower()}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Guardado: ig_kmers_{label_str.lower()}.png")
    return top_kmers

# ── CORRER ───────────────────────────────────────────────────
idx_R = np.where(labels_muestra == 1)[0][0]
idx_S = np.where(labels_muestra == 0)[0][0]

print("🧬 Integrated Gradients 6-mer — Resistente...")
motivos_R = integrated_gradients_motifs(secuencias[idx_R], "Resistant", target_class=1)

print("\n🧬 Integrated Gradients 6-mer — Susceptible...")
motivos_S = integrated_gradients_motifs(secuencias[idx_S], "Susceptible", target_class=0)

============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (4).ipynb Cell 29 ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Resistente
nombres_R = [k[0] for k in motivos_R[:15]]
valores_R = [k[1] for k in motivos_R[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("Resistant sequence\nTop 15 6-mer motifs", fontsize=12)

# Susceptible
nombres_S = [k[0] for k in motivos_S[:15]]
valores_S = [k[1] for k in motivos_S[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle("Figure X — Integrated Gradients 6-mer Motif Analysis\n"
             "NTv3 attribution scores for Ertapenem Resistance vs Susceptibility",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_IG_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X guardada")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (4).ipynb Cell 40 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask",
                    (input_ids != tokenizer.pad_token_id).long()).to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = (baseline_embeds + alpha *
                       (input_embeds - baseline_embeds)).detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads *
                   (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)
    kmer_scores = {}

    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        motivo = seq_limpia[i:i+kmer]
        if len(motivo) == kmer and all(c in 'ATGC' for c in motivo):
            score = float(np.mean(token_importance[i:i+kmer]))
            if motivo in kmer_scores:
                kmer_scores[motivo] = max(kmer_scores[motivo], score)
            else:
                kmer_scores[motivo] = score

    return sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("✅ Función integrated_gradients_motifs definida")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (5).ipynb Cell 18 ===
import torch
import numpy as np
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

def analizar_interpretabilidad_mil(model, tokenizer, bag, device="cuda"):
    model.to(device)
    model.eval()

    sequences = bag["sequences"]
    genome_id = bag.get("genome_id", "Unknown")
    label = bag.get("label", "?")

    print(f"🔍 Analizando Genoma: {genome_id} (Label: {label})")

    # --- PASO 1: Identificar el Gen con mayor Probabilidad (Top-1) ---
    all_probs = []
    sub_batch = 8

    with torch.no_grad():
        for start in range(0, len(sequences), sub_batch):
            chunk = sequences[start : start + sub_batch]
            inputs = tokenizer(chunk, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
            sp_ids = torch.full((inputs['input_ids'].size(0),), 6, device=device)

            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'), species_ids=sp_ids)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs)

    all_probs = torch.cat(all_probs)
    top_instance_idx = torch.argmax(all_probs).item()
    top_sequence = sequences[top_instance_idx]

    print(f"📍 Gen detectado como 'Driver' de Resistencia: {top_sequence[:60]}...")
    print(f"🔥 Probabilidad del Gen: {all_probs[top_instance_idx].item():.4f}")

    # --- PASO 2: Integrated Gradients sobre el Gen Ganador ---
    def predict_wrapper(inputs_embeds):
        # FIX: Expand species_ids to match the batch size created by Captum steps
        batch_size = inputs_embeds.shape[0]
        sp_id = torch.full((batch_size,), 6, device=device)

        outputs = model.ntv3(inputs_embeds=inputs_embeds, species_ids=sp_id, output_hidden_states=True)
        last_hidden_layer = outputs.hidden_states[-1]
        cls_output = last_hidden_layer[:, 0, :]
        logits = model.classifier(model.dropout(cls_output))
        return logits[:, 1]

    ig = IntegratedGradients(predict_wrapper)

    inputs_top = tokenizer(top_sequence, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs_top.input_ids

    with torch.no_grad():
        input_embeds = model.ntv3.core.embed_layer(input_ids)

    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    baseline_embeds = model.ntv3.core.embed_layer(baseline_ids)

    # Run attribution
    attributions = ig.attribute(inputs=input_embeds, baselines=baseline_embeds, target=None)

    scores = attributions.squeeze(0).norm(p=2, dim=-1).cpu().detach().numpy()

    # --- PASO 3: Visualización ---
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(scores)), scores, color='crimson', alpha=0.7)
    plt.title(f"Importancia (Integrated Gradients) - Gen Driver en {genome_id}")
    plt.xlabel("Posición del Token / Nucleótido")
    plt.ylabel("Atribución")
    plt.grid(alpha=0.2)
    plt.show()

    return top_sequence, scores, tokenizer.convert_ids_to_tokens(input_ids[0])
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (5).ipynb Cell 28 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        # CORRECCIÓN: Usar core.embed_layer en lugar de embeddings
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated = interpolated.detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    # ── MAPEAR ATRIBUCIONES A KMERS DE LA SECUENCIA ORIGINAL ─
    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)

    # Construir kmers con sus atribuciones
    kmer_scores = {}
    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        kmer_seq = seq_limpia[i:i+kmer]
        score = float(np.mean(token_importance[i:i+kmer]))
        if kmer_seq in kmer_scores:
            kmer_scores[kmer_seq] += score
        else:
            kmer_scores[kmer_seq] = score

    # Top 20 kmers por score total
    top_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\nTop 10 motivos 6-mer ({label_str}):")
    for km, sc in top_kmers[:10]:
        print(f"  {km}: {sc:.6f}")

    # ── FIGURA ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    nombres = [k[0] for k in top_kmers]
    valores = [k[1] for k in top_kmers]
    color = 'red' if label_str == 'Resistant' else 'blue'
    plt.barh(nombres, valores, color=color, alpha=0.85)
    plt.xlabel("Integrated Gradients Attribution (mean over k-mer positions)")
    plt.title(f"Figure — Integrated Gradients 6-mer Motifs: {label_str} sequence\n"
              f"(top 20 motifs by attribution score)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ruta = f'{RUTA_BASE}/ig_kmers_{label_str.lower()}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Guardado: ig_kmers_{label_str.lower()}.png")
    return top_kmers

# ── CORRER ───────────────────────────────────────────────────
idx_R = np.where(labels_muestra == 1)[0][0]
idx_S = np.where(labels_muestra == 0)[0][0]

print("🧬 Integrated Gradients 6-mer — Resistente...")
motivos_R = integrated_gradients_motifs(secuencias[idx_R], "Resistant", target_class=1)

print("\n🧬 Integrated Gradients 6-mer — Susceptible...")
motivos_S = integrated_gradients_motifs(secuencias[idx_S], "Susceptible", target_class=0)

============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (5).ipynb Cell 29 ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Resistente
nombres_R = [k[0] for k in motivos_R[:15]]
valores_R = [k[1] for k in motivos_R[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("Resistant sequence\nTop 15 6-mer motifs", fontsize=12)

# Susceptible
nombres_S = [k[0] for k in motivos_S[:15]]
valores_S = [k[1] for k in motivos_S[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle("Figure X — Integrated Gradients 6-mer Motif Analysis\n"
             "NTv3 attribution scores for Ertapenem Resistance vs Susceptibility",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_IG_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X guardada")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil) (5).ipynb Cell 40 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask",
                    (input_ids != tokenizer.pad_token_id).long()).to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = (baseline_embeds + alpha *
                       (input_embeds - baseline_embeds)).detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads *
                   (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)
    kmer_scores = {}

    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        motivo = seq_limpia[i:i+kmer]
        if len(motivo) == kmer and all(c in 'ATGC' for c in motivo):
            score = float(np.mean(token_importance[i:i+kmer]))
            if motivo in kmer_scores:
                kmer_scores[motivo] = max(kmer_scores[motivo], score)
            else:
                kmer_scores[motivo] = score

    return sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("✅ Función integrated_gradients_motifs definida")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil).ipynb Cell 18 ===
import torch
import numpy as np
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

def analizar_interpretabilidad_mil(model, tokenizer, bag, device="cuda"):
    model.to(device)
    model.eval()

    sequences = bag["sequences"]
    genome_id = bag.get("genome_id", "Unknown")
    label = bag.get("label", "?")

    print(f"🔍 Analizando Genoma: {genome_id} (Label: {label})")

    # --- PASO 1: Identificar el Gen con mayor Probabilidad (Top-1) ---
    all_probs = []
    sub_batch = 8

    with torch.no_grad():
        for start in range(0, len(sequences), sub_batch):
            chunk = sequences[start : start + sub_batch]
            inputs = tokenizer(chunk, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
            sp_ids = torch.full((inputs['input_ids'].size(0),), 6, device=device)

            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'), species_ids=sp_ids)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs)

    all_probs = torch.cat(all_probs)
    top_instance_idx = torch.argmax(all_probs).item()
    top_sequence = sequences[top_instance_idx]

    print(f"📍 Gen detectado como 'Driver' de Resistencia: {top_sequence[:60]}...")
    print(f"🔥 Probabilidad del Gen: {all_probs[top_instance_idx].item():.4f}")

    # --- PASO 2: Integrated Gradients sobre el Gen Ganador ---
    def predict_wrapper(inputs_embeds):
        # FIX: Expand species_ids to match the batch size created by Captum steps
        batch_size = inputs_embeds.shape[0]
        sp_id = torch.full((batch_size,), 6, device=device)

        outputs = model.ntv3(inputs_embeds=inputs_embeds, species_ids=sp_id, output_hidden_states=True)
        last_hidden_layer = outputs.hidden_states[-1]
        cls_output = last_hidden_layer[:, 0, :]
        logits = model.classifier(model.dropout(cls_output))
        return logits[:, 1]

    ig = IntegratedGradients(predict_wrapper)

    inputs_top = tokenizer(top_sequence, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs_top.input_ids

    with torch.no_grad():
        input_embeds = model.ntv3.core.embed_layer(input_ids)

    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    baseline_embeds = model.ntv3.core.embed_layer(baseline_ids)

    # Run attribution
    attributions = ig.attribute(inputs=input_embeds, baselines=baseline_embeds, target=None)

    scores = attributions.squeeze(0).norm(p=2, dim=-1).cpu().detach().numpy()

    # --- PASO 3: Visualización ---
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(scores)), scores, color='crimson', alpha=0.7)
    plt.title(f"Importancia (Integrated Gradients) - Gen Driver en {genome_id}")
    plt.xlabel("Posición del Token / Nucleótido")
    plt.ylabel("Atribución")
    plt.grid(alpha=0.2)
    plt.show()

    return top_sequence, scores, tokenizer.convert_ids_to_tokens(input_ids[0])
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil).ipynb Cell 28 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        # CORRECCIÓN: Usar core.embed_layer en lugar de embeddings
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interpolated = interpolated.detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    # ── MAPEAR ATRIBUCIONES A KMERS DE LA SECUENCIA ORIGINAL ─
    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)

    # Construir kmers con sus atribuciones
    kmer_scores = {}
    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        kmer_seq = seq_limpia[i:i+kmer]
        score = float(np.mean(token_importance[i:i+kmer]))
        if kmer_seq in kmer_scores:
            kmer_scores[kmer_seq] += score
        else:
            kmer_scores[kmer_seq] = score

    # Top 20 kmers por score total
    top_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\nTop 10 motivos 6-mer ({label_str}):")
    for km, sc in top_kmers[:10]:
        print(f"  {km}: {sc:.6f}")

    # ── FIGURA ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    nombres = [k[0] for k in top_kmers]
    valores = [k[1] for k in top_kmers]
    color = 'red' if label_str == 'Resistant' else 'blue'
    plt.barh(nombres, valores, color=color, alpha=0.85)
    plt.xlabel("Integrated Gradients Attribution (mean over k-mer positions)")
    plt.title(f"Figure — Integrated Gradients 6-mer Motifs: {label_str} sequence\n"
              f"(top 20 motifs by attribution score)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ruta = f'{RUTA_BASE}/ig_kmers_{label_str.lower()}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Guardado: ig_kmers_{label_str.lower()}.png")
    return top_kmers

# ── CORRER ───────────────────────────────────────────────────
idx_R = np.where(labels_muestra == 1)[0][0]
idx_S = np.where(labels_muestra == 0)[0][0]

print("🧬 Integrated Gradients 6-mer — Resistente...")
motivos_R = integrated_gradients_motifs(secuencias[idx_R], "Resistant", target_class=1)

print("\n🧬 Integrated Gradients 6-mer — Susceptible...")
motivos_S = integrated_gradients_motifs(secuencias[idx_S], "Susceptible", target_class=0)

============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil).ipynb Cell 29 ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Resistente
nombres_R = [k[0] for k in motivos_R[:15]]
valores_R = [k[1] for k in motivos_R[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("Resistant sequence\nTop 15 6-mer motifs", fontsize=12)

# Susceptible
nombres_S = [k[0] for k in motivos_S[:15]]
valores_S = [k[1] for k in motivos_S[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle("Figure X — Integrated Gradients 6-mer Motif Analysis\n"
             "NTv3 attribution scores for Ertapenem Resistance vs Susceptibility",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_IG_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X guardada")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil).ipynb Cell 35 ===
# IG sobre KPC-3 real
print("🧬 Integrated Gradients — KPC-3 (carbapenemasa clínica)...")
motivos_KPC3 = integrated_gradients_motifs(seq_kpc3, "KPC-3 Resistant", target_class=1)

# Buscar también una secuencia susceptible con gen bien identificado
# para comparación limpia
seq_sus = val_df[val_df['label'] == 0].iloc[0]['sequence']
print("\n🧬 Integrated Gradients — Susceptible...")
motivos_SUS = integrated_gradients_motifs(seq_sus, "Susceptible", target_class=0)

# Figura comparativa final
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

nombres_R = [k[0] for k in motivos_KPC3[:15]]
valores_R = [k[1] for k in motivos_KPC3[:15]]
axes[0].barh(nombres_R, valores_R, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("IG Attribution Score")
axes[0].set_title("KPC-3 beta-lactamase (Resistant)\nTop 15 6-mer motifs", fontsize=12)

nombres_S = [k[0] for k in motivos_SUS[:15]]
valores_S = [k[1] for k in motivos_SUS[:15]]
axes[1].barh(nombres_S, valores_S, color='blue', alpha=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel("IG Attribution Score")
axes[1].set_title("Susceptible sequence\nTop 15 6-mer motifs", fontsize=12)

plt.suptitle(
    "Figure X — Integrated Gradients 6-mer Motif Analysis\n"
    "KPC-3 beta-lactamase vs Susceptible: NTv3 attribution scores for Ertapenem resistance",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(f'{RUTA_BASE}/figura_X_KPC3_vs_susceptible.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura X final guardada: figura_X_KPC3_vs_susceptible.png")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil).ipynb Cell 47 ===
def integrated_gradients_motifs(secuencia, label_str, target_class=1, steps=50, kmer=6):
    modelo.eval()

    enc = tokenizer(
        secuencia,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask",
                    (input_ids != tokenizer.pad_token_id).long()).to(device)
    sp_id = torch.tensor([6]).to(device)

    with torch.no_grad():
        input_embeds = modelo.ntv3.core.embed_layer(input_ids)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline_embeds = modelo.ntv3.core.embed_layer(baseline_ids)

    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(steps):
        alpha = step / steps
        interpolated = (baseline_embeds + alpha *
                       (input_embeds - baseline_embeds)).detach().requires_grad_(True)
        outputs = modelo.ntv3(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
            species_ids=sp_id,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        logits = modelo.classifier(modelo.dropout(cls_output))
        prob = torch.softmax(logits, dim=1)[0, target_class]
        prob.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / steps
    attributions = (integrated_grads *
                   (input_embeds - baseline_embeds)).detach()
    token_importance = attributions[0].norm(dim=-1).cpu().numpy()

    seq_limpia = secuencia[:512]
    n_tokens = len(token_importance)
    kmer_scores = {}

    for i in range(min(n_tokens - kmer, len(seq_limpia) - kmer)):
        motivo = seq_limpia[i:i+kmer]
        if len(motivo) == kmer and all(c in 'ATGC' for c in motivo):
            score = float(np.mean(token_importance[i:i+kmer]))
            if motivo in kmer_scores:
                kmer_scores[motivo] = max(kmer_scores[motivo], score)
            else:
                kmer_scores[motivo] = score

    return sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("✅ Función integrated_gradients_motifs definida")
============================================================
=== Found in Flaming_AI(produccion_a_nivel_de_genoma_secuencias_v3_con_mil).ipynb Cell 48 ===
# ── IG SOBRE LAS 25 KPC-3 ────────────────────────────────────
print("Corriendo IG sobre 25 secuencias KPC-3...")

todos_motivos = {}

for idx, row in tqdm(df_kpc3.iterrows(), total=len(df_kpc3), desc="IG KPC-3"):
    try:
        motivos = integrated_gradients_motifs(
            row['sequence'],
            label_str="KPC-3",
            target_class=1,
            steps=50,
            kmer=6
        )
        for motivo, score in motivos:
            if motivo not in todos_motivos:
                todos_motivos[motivo] = []
            todos_motivos[motivo].append(score)
    except Exception as e:
        print(f"Error en secuencia {idx}: {e}")
        continue

# ── AGREGAR POR MOTIVO ───────────────────────────────────────
motivos_agregados = {
    motivo: np.mean(scores)
    for motivo, scores in todos_motivos.items()
    if len(scores) >= 5  # aparece en al menos 5 de 25 secuencias
}

top_motivos = sorted(motivos_agregados.items(), key=lambda x: x[1], reverse=True)[:20]

print(f"\n✅ Motivos únicos encontrados: {len(todos_motivos)}")
print(f"Motivos consistentes (≥5 secuencias): {len(motivos_agregados)}")
print(f"\nTop 10 motivos universales KPC-3:")
for motivo, score in top_motivos[:10]:
    n = len(todos_motivos[motivo])
    print(f"  {motivo}: {score:.4f} (presente en {n}/25 secuencias)")

# ── FIGURA ───────────────────────────────────────────────────
nombres = [m[0] for m in top_motivos]
valores = [m[1] for m in top_motivos]
frecuencias = [len(todos_motivos[m[0]]) for m in top_motivos]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1 — Score promedio
axes[0].barh(nombres, valores, color='red', alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_xlabel("Mean IG Attribution Score (n=25 sequences)")
axes[0].set_title("Top 20 Universal KPC-3 Motifs\nMean IG score across 25 sequences", fontsize=11)

# Plot 2 — Frecuencia de aparición
axes[1].barh(nombres, frecuencias, color='darkred', alpha=0.85)
axes[1].invert_yaxis()
axes[1].axvline(x=25, color='black', linestyle='--', alpha=0.5, label='n=25')
axes[1].set_xlabel("Number of sequences containing motif (out of 25)")
axes[1].set_title("Motif Consistency\nHow many KPC-3 sequences share each motif", fontsize=11)
axes[1].legend()

plt.suptitle(
    "Figure — Population-Level Integrated Gradients Analysis\n"
    "Universal resistance motifs across 25 KPC-3 beta-lactamase sequences",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    '/content/drive/MyDrive/rgi_Klebsiella_pneumoniae_ertapenem/dataset_limpio/ig_kpc3_population.png',
    dpi=150, bbox_inches='tight'
)
plt.show()
print("✅ Guardado: ig_kpc3_population.png")
============================================================