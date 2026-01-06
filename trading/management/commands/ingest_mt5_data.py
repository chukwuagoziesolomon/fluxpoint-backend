from django.core.management.base import BaseCommand
from trading.mt5_integration import MT5DataIngestion
import os

class Command(BaseCommand):
    help = 'Ingest historical data from MT5 terminal'

    def add_arguments(self, parser):
        parser.add_argument(
            '--symbols',
            nargs='+',
            type=str,
            default=['EURUSD'],
            help='List of symbols to ingest, e.g., EURUSD GBPUSD'
        )
        parser.add_argument(
            '--timeframes',
            nargs='+',
            type=str,
            default=['M15', 'H1', 'D1'],
            help='List of timeframes to ingest, e.g., M15 H1 D1'
        )
        parser.add_argument(
            '--days-back',
            type=int,
            default=30,
            help='Number of days back to fetch data'
        )
        parser.add_argument(
            '--login',
            type=str,
            help='MT5 login'
        )
        parser.add_argument(
            '--password',
            type=str,
            help='MT5 password'
        )
        parser.add_argument(
            '--server',
            type=str,
            help='MT5 server'
        )
        parser.add_argument(
            '--mode',
            type=str,
            choices=['incremental', 'full'],
            default='incremental',
            help='Ingestion mode: incremental (from latest) or full (from days_back)'
        )

    def handle(self, *args, **options):
        # Get credentials from environment or arguments
        login = options['login'] or os.getenv('MT5_LOGIN')
        password = options['password'] or os.getenv('MT5_PASSWORD')
        server = options['server'] or os.getenv('MT5_SERVER')

        ingestor = MT5DataIngestion(login=login, password=password, server=server)

        if not ingestor.connect():
            self.stderr.write("Failed to connect to MT5")
            return

        try:
            ingestor.ingest_data(
                symbols=options['symbols'],
                timeframes=options['timeframes'],
                mode=options['mode'],
                days_back=options['days_back']
            )
            self.stdout.write("Data ingestion completed")
        finally:
            ingestor.disconnect()