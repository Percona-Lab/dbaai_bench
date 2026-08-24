"""Where the model comes from: gateway, credential, catalog and prices.

Three gateways are supported - two hosted, one you run yourself - and all speak
the OpenAI wire format, so one client serves any of them. What differs between
them is the base URL, which environment variable holds the key (or whether there
is one at all), how model ids are spelled, and whether the gateway publishes its
own rates or bills anything - all of which is in providers.py.

What they say about their own models differs too. A hosted gateway describes them
in the same /v1/models response the catalog is built from; a self-hosted one
answers with three fields and keeps the rest - context length, whether the weights
are in memory - on an endpoint of its own, which details.py reads where there is
one and shrugs off where there is not.

Nothing above this package knows which gateway is in use: cli.py picks a
Provider, and the rest of the harness is handed a client and a PriceBook.
"""
