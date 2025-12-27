# Guide de Configuration Twilio WhatsApp

## Étape 1: Créer un Compte Twilio

1. Allez sur https://www.twilio.com/try-twilio
2. Créez un compte gratuit (trial account)
3. Vérifiez votre email et numéro de téléphone

## Étape 2: Obtenir vos Identifiants

1. Une fois connecté, allez sur le Dashboard: https://console.twilio.com/
2. Vous verrez:
   - **Account SID** (commence par AC...)
   - **Auth Token** (cliquez sur "Show" pour le révéler)
3. Copiez ces deux valeurs

## Étape 3: Configurer WhatsApp Sandbox

### Option A: Utiliser le Sandbox (Gratuit - Recommandé pour commencer)

1. Dans la console Twilio, allez dans **Messaging** → **Try it out** → **Send a WhatsApp message**
2. Vous verrez:
   - Le numéro WhatsApp de Twilio (ex: `+1 415 523 8886`)
   - Un code à envoyer (ex: `join <votre-code>`)

3. **Sur votre téléphone:**
   - Ouvrez WhatsApp
   - Créez un nouveau message pour le numéro: `+1 415 523 8886`
   - Envoyez le message: `join <votre-code>` (remplacez par le code affiché)
   - Vous recevrez une confirmation

4. **Pour chaque contact qui doit recevoir des messages:**
   - Ils doivent également envoyer `join <votre-code>` au même numéro
   - C'est une limitation du sandbox (mode test)

### Option B: Numéro WhatsApp Approuvé (Payant - Production)

Pour un usage sans limitations:
1. Demandez l'approbation d'un numéro WhatsApp Business
2. Processus peut prendre plusieurs jours
3. Coûts: ~$0.005 par message

## Étape 4: Configurer config.py

Ouvrez votre `config.py` et modifiez:

```python
# Twilio WhatsApp
TWILIO_ACCOUNT_SID = "AC1234567890abcdef..."  # Collez votre Account SID
TWILIO_AUTH_TOKEN = "votre_auth_token_ici"    # Collez votre Auth Token
TWILIO_WHATSAPP_FROM = "whatsapp:+14155238886"  # Numéro Twilio (sandbox ou approuvé)

# Contacts WhatsApp
WHATSAPP_CONTACTS = {
    "moi": "whatsapp:+33612345678",      # Votre numéro (format: +code_pays puis numéro)
    "marie": "whatsapp:+33687654321",    # Exemple
    "papa": "whatsapp:+33698765432",     # Ajoutez vos contacts
}
```

**Important pour les numéros:**
- Format: `whatsapp:+code_pays_numéro`
- France: `whatsapp:+336...` ou `whatsapp:+337...`
- Pas d'espaces, pas de tirets
- Commencez par le code pays (+33 pour France)

## Étape 5: Installer Twilio

```bash
py -m pip install twilio
```

## Étape 6: Tester

Lancez l'assistant:
```bash
py voice_assistant.py
```

Essayez:
- "Envoie un message WhatsApp à moi: test de l'assistant"
- "Envoie un WhatsApp à Marie: je serai en retard"
- "Envoie-moi un message: acheter du lait"

## Commandes Vocales Supportées

- **"Envoie un message WhatsApp à [contact]: [message]"**
- **"Envoie un WhatsApp à [contact]: [message]"**
- **"Envoie-moi un message: [message]"** (utilise le contact "moi")

Exemples:
- "Envoie un WhatsApp à papa: j'arrive dans 10 minutes"
- "Envoie-moi un message: rappeler d'appeler le dentiste"
- "Envoie un message à Marie: rendez-vous confirmé pour demain"

## Dépannage

### "Contact non trouvé"
- Vérifiez que le nom du contact est exactement comme dans `WHATSAPP_CONTACTS`
- Les noms sont en minuscules (automatiquement converti)
- Utilisez `moi`, `marie`, etc. sans accents

### "Failed to send message"
- Vérifiez que le destinataire a bien envoyé `join <code>` (sandbox uniquement)
- Vérifiez vos identifiants Twilio
- Vérifiez le format du numéro: `whatsapp:+33...`

### "Twilio non configuré"
- Vérifiez que `TWILIO_ACCOUNT_SID` et `TWILIO_AUTH_TOKEN` sont bien renseignés
- Pas de guillemets supplémentaires
- Pas d'espaces avant/après

### Le message ne s'envoie pas
- Vérifiez votre crédit Twilio (compte trial a des limitations)
- Le sandbox expire après 3 jours d'inactivité - renvoyez `join <code>`
- Consultez les logs Twilio: https://console.twilio.com/monitor/logs/messages

## Limitations du Sandbox (Gratuit)

- Les destinataires doivent d'abord rejoindre avec `join <code>`
- Le sandbox expire après 3 jours sans activité
- Le code `join` doit être renvoyé pour réactiver
- Nombre limité de messages

## Passer en Production

Pour supprimer ces limitations:
1. Dans Twilio Console → WhatsApp → Senders
2. Demandez l'approbation d'un numéro
3. Remplissez le formulaire (nom d'entreprise, cas d'usage)
4. Attendez l'approbation (quelques jours)
5. Mettez à jour `TWILIO_WHATSAPP_FROM` avec votre nouveau numéro

## Coûts

**Sandbox (Test):** Gratuit
**Production:** 
- ~$0.005 par message (varie selon le pays)
- Crédit gratuit de $15-20 pour nouveaux comptes
- Pas d'abonnement mensuel

## Support

- Documentation Twilio: https://www.twilio.com/docs/whatsapp
- Console Twilio: https://console.twilio.com/
- Support: https://support.twilio.com/
