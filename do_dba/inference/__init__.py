"""Where the model comes from: gateway, credential, catalog and prices.

Two gateways are supported and both speak the OpenAI wire format, so one client
serves either. What differs between them is the base URL, which environment
variable holds the key, how model ids are spelled, and whether the gateway
publishes its own rates - all of which is in providers.py.

Nothing above this package knows which gateway is in use: cli.py picks a
Provider, and the rest of the harness is handed a client and a PriceBook.
"""
