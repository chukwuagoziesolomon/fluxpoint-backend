from django.core.management.base import BaseCommand
from trading.mt5_integration import MT5DataIngestion
import os
import logging
from apscheduler.schedulers.blocking import BlockingScheduler

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Schedule automatic MT5 data ingestion'

    def add_arguments(self, parser):
        parser.add_argument(
            '--symbols',
            nargs='+',
            type=str,
            default=['EURUSD'],
            help='List of symbols to ingest'
        )
        parser.add_argument(
            '--timeframes',
            nargs='+',
            type=str,
            default=['M15', 'H1', 'D1'],
            help='List of timeframes to ingest'
        )
        parser.add_argument(
            '--interval-hours',
            type=int,
            default=1,
            help='Interval in hours for scheduled ingestion'
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

    def handle(self, *args, **options):
        login = options['login'] or os.getenv('MT5_LOGIN')
        password = options['password'] or os.getenv('MT5_PASSWORD')
        server = options['server'] or os.getenv('MT5_SERVER')

        ingestor = MT5DataIngestion(login=login, password=password, server=server)

        symbols = options['symbols']
        timeframes = options['timeframes']
        interval_hours = options['interval_hours']

        def run_ingestion():
            self.stdout.write("Starting scheduled ingestion...")
            if not ingestor.connect():
                self.stderr.write("Failed to connect to MT5")
                return
            try:
                ingestor.ingest_data(symbols, timeframes, mode='incremental')
                self.stdout.write("Scheduled ingestion completed")
            except Exception as e:
                logger.error(f"Error during scheduled ingestion: {e}")
                self.stderr.write(f"Error: {e}")
            finally:
                ingestor.disconnect()

        scheduler = BlockingScheduler()
        scheduler.add_job(run_ingestion, 'interval', hours=interval_hours)
        self.stdout.write(f"Starting scheduler with {interval_hours} hour intervals")
        try:
            scheduler.start()
        except KeyboardInterrupt:
            self.stdout.write("Scheduler stopped")