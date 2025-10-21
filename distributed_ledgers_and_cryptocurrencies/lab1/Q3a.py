from sys import exit
from bitcoin.core.script import *
from bitcoin.wallet import CBitcoinSecret

from lib.utils import *
from lib.config import (my_private_key, my_public_key, my_address,
                        faucet_address, network_type)
from Q1 import send_from_P2PKH_transaction


# TODO: Generirajte privatne kljuceve od klijenata koristeci `lib/keygen.py`
# i dodajte ih ovdje.
cust1_private_key = CBitcoinSecret('cMzVS2Nmmjx31nWXd8P7uUkDkYGE3ST9KBMrcQ6hnK8sYmXGwy7m')
cust1_public_key = cust1_private_key.pub

cust2_private_key = CBitcoinSecret('cNpBf8D47iyzvzszHXPMZa4yN7cEXUr3gyDZHyGvnf814b7ec2Mh')
cust2_public_key = cust2_private_key.pub

cust3_private_key = CBitcoinSecret('cVkcnqFpVYH7U3nEhfcw6o4oLECrJUaGYHNYe1KPyGTH9boAfM3f')
cust3_public_key = cust3_private_key.pub

bank_public_key = my_public_key

######################################################################
# TODO: Implementirajte `scriptPubKey` za zadatak 3

# Pretpostavite da vi igrate ulogu banke u ovom zadatku na nacin da privatni
# kljuc od banke `bank_private_key` odgovara vasem privatnom kljucu
# `my_private_key`.
Q3a_txout_scriptPubKey = [
    bank_public_key,    
    OP_CHECKSIGVERIFY,  
    OP_1,               
    cust1_public_key,   
    cust2_public_key,   
    cust3_public_key,   
    OP_3,               
    OP_CHECKMULTISIG    
]
######################################################################

if __name__ == '__main__':
    ######################################################################
    # Postavite parametre transakcije
    # TODO: amount_to_send = {cjelokupni iznos BCY-a u UTXO-u kojeg otkljucavamo} - {fee}
    amount_to_send = 0.00015
    # TODO: Identifikator transakcije
    txid_to_spend = (
        'de89ce2d58f50c449e16f494cdc3106dec0e01f84343c58f434814239cc795cc')
    # TODO: indeks UTXO-a unutar transakcije na koju se referiramo
    # (indeksi pocinju od nula)
    utxo_index = 2
    ######################################################################

    response = send_from_P2PKH_transaction(amount_to_send, txid_to_spend,
                                           utxo_index, Q3a_txout_scriptPubKey,
                                           my_private_key, network_type)
    print(response.status_code, response.reason)
    print(response.text)
