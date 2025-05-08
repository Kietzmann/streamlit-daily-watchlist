# worker_tasks.py (Corrected and Revised for 1m/5m/15m)
import asyncio
import ccxt.async_support as ccxt
import logging
import os
import time
import numpy as np # Keep numpy for potential future use or error checking

# Configure logger specifically for worker processes
logging.basicConfig(
    level=logging.INFO, # Change to DEBUG for extensive logs
    format="%(asctime)s - %(levelname)s - [Worker:%(process)d] - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Use a specific logger name
worker_logger = logging.getLogger(f"worker.{os.getpid()}")
worker_logger.propagate = False # Avoid double logging if main has root logger configured

# Define required data lengths (adjust based on indicator needs, e.g., breakout lookback)
# Make them generous to ensure indicators can calculate
FETCH_LIMIT_1M = 30   # Needs at least 2 for change
FETCH_LIMIT_5M = 40   # Needs ~13 for 12-bar volatility + 1 for change
FETCH_LIMIT_15M = 50  # Needs BREAKOUT_LOOKBACK + smoothing window (e.g., 24 + 6 = 30+)

async def fetch_symbol_short_term_data(exchange, symbol):
    """
    Fetches 1m, 5m, and 15m OHLCV data concurrently for a single symbol.
    Returns results including the fetched data and latest price.
    """
    results = {
        "symbol": symbol,
        "ohlcv_1m": None,
        "ohlcv_5m": None,
        "ohlcv_15m": None,
        "price": None, # Store latest price from shortest timeframe
        "error": None
    }
    fetch_errors = []

    try:
        worker_logger.debug(f"Fetching short-term data for {symbol}...")

        # Create tasks for each timeframe fetch
        tasks = {
            '1m': exchange.fetch_ohlcv(symbol, timeframe='1m', limit=FETCH_LIMIT_1M),
            '5m': exchange.fetch_ohlcv(symbol, timeframe='5m', limit=FETCH_LIMIT_5M),
            '15m': exchange.fetch_ohlcv(symbol, timeframe='15m', limit=FETCH_LIMIT_15M),
        }

        # Run fetches concurrently
        task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Process results from gather
        ohlcv_data = {}
        for i, tf in enumerate(tasks.keys()):
            res = task_results[i]
            if isinstance(res, Exception):
                error_msg = f"Fetch {tf} failed: {type(res).__name__} - {res}"
                worker_logger.warning(f"{symbol}: {error_msg}")
                fetch_errors.append(error_msg)
                ohlcv_data[tf] = None # Store None on error
            elif isinstance(res, list) and len(res) > 0:
                 ohlcv_data[tf] = res # Store successful OHLCV list
                 # Get latest price from the 1m data if available (most recent)
                 if tf == '1m' and res[-1] and len(res[-1]) >= 5:
                     try:
                         results["price"] = float(res[-1][4]) # Close price of last 1m candle
                     except (TypeError, ValueError, IndexError):
                          worker_logger.warning(f"{symbol}: Could not parse price from last 1m candle: {res[-1:]}")
            elif isinstance(res, list) and len(res) == 0:
                 error_msg = f"Fetch {tf} returned empty list"
                 worker_logger.warning(f"{symbol}: {error_msg}")
                 fetch_errors.append(error_msg)
                 ohlcv_data[tf] = None
            else:
                 error_msg = f"Fetch {tf} returned unexpected type: {type(res)}"
                 worker_logger.warning(f"{symbol}: {error_msg}")
                 fetch_errors.append(error_msg)
                 ohlcv_data[tf] = None

        # Assign fetched data to results dictionary
        results["ohlcv_1m"] = ohlcv_data.get('1m')
        results["ohlcv_5m"] = ohlcv_data.get('5m')
        results["ohlcv_15m"] = ohlcv_data.get('15m')

        # If price wasn't set from 1m, try 5m then 15m
        if results["price"] is None:
            for tf_key in ["ohlcv_5m", "ohlcv_15m"]:
                 ohlcv_list = results.get(tf_key)
                 if ohlcv_list and ohlcv_list[-1] and len(ohlcv_list[-1]) >= 5:
                      try:
                           results["price"] = float(ohlcv_list[-1][4])
                           break # Stop once price is found
                      except (TypeError, ValueError, IndexError):
                           continue

        # Combine fetch errors if any occurred
        if fetch_errors:
            results["error"] = "; ".join(fetch_errors)

        worker_logger.debug(f"Finished fetching short-term data for {symbol}. Price: {results['price']}. Error: {results['error']}")
        return results

    # Catch broader exceptions during the process for this symbol
    except ccxt.BadSymbol as e:
        msg=f"BadSymbol: {e}"
        worker_logger.debug(f"OHLCV fetch failed for {symbol}: {msg}")
        results["error"]=msg
        return results
    except ccxt.NetworkError as e:
        msg=f"NetworkError: {e}"
        worker_logger.warning(f"OHLCV fetch failed for {symbol}: {msg}")
        results["error"]=msg
        return results
    except ccxt.ExchangeError as e:
        msg=f"ExchangeError: {e}"
        worker_logger.warning(f"OHLCV fetch failed for {symbol}: {msg}")
        results["error"]=msg
        return results
    except asyncio.TimeoutError:
        msg="Request Timeout during gather/fetch"
        worker_logger.warning(f"OHLCV fetch failed for {symbol}: {msg}")
        results["error"]=msg
        return results
    except Exception as e:
        msg=f"Unexpected Error in fetch_symbol_short_term_data for {symbol}: {type(e).__name__} - {e}"
        worker_logger.error(msg, exc_info=True)
        results["error"] = msg
        return results


# --- Multiprocessing Target Function ---
def process_symbol_batch(exchange_id, symbols, timeframe, limit, ccxt_timeout):
    """
    Synchronous function executed in a separate process.
    Sets up an asyncio event loop and runs fetch_symbol_short_term_data for a batch.
    'timeframe' and 'limit' arguments are ignored as specific TFs are fetched internally.
    """
    worker_logger.info(f"Worker {os.getpid()} starting process for {len(symbols)} symbols.")

    async def fetch_batch_async():
        exchange = None
        results = []
        try:
            # Configure exchange instance within the async function
            exchange_config = {
                "enableRateLimit": True,
                "options": {'fetchOHLCVWarning': False}, # Suppress ccxt warning if limit > max
                "timeout": ccxt_timeout # Use the provided timeout
            }
            exchange = getattr(ccxt, exchange_id)(exchange_config)

            # Create tasks for fetching data for each symbol in the batch
            tasks = [fetch_symbol_short_term_data(exchange, symbol) for symbol in symbols]

            # Run tasks concurrently and gather results
            # return_exceptions=True ensures gather doesn't stop on first error
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results, handling potential exceptions returned by gather
            for i, res in enumerate(task_results):
                symbol = symbols[i] # Get corresponding symbol
                if isinstance(res, Exception):
                    # This catches exceptions *during* the gather/task execution itself (less common)
                    worker_logger.warning(f"Exception during task execution for {symbol}: {type(res).__name__} - {res}")
                    # Append error information for this symbol
                    results.append({
                        "symbol": symbol, "ohlcv_1m": None, "ohlcv_5m": None, "ohlcv_15m": None,
                        "price": None, "error": f"Worker Task Exception: {type(res).__name__} - {res}"
                    })
                elif isinstance(res, dict):
                    # Append successful result dictionary (which might contain its own 'error' key from fetching)
                    results.append(res)
                else:
                    # Handle unexpected result types from fetch_symbol_short_term_data
                    worker_logger.error(f"Unexpected result type ({type(res)}) for {symbol} in worker.")
                    results.append({
                        "symbol": symbol, "ohlcv_1m": None, "ohlcv_5m": None, "ohlcv_15m": None,
                        "price": None, "error": "Unexpected worker result type"
                    })
            worker_logger.info(f"Worker {os.getpid()} finished batch processing {len(symbols)} symbols.")
            return results

        except Exception as e:
             # Catch errors during exchange setup or critical asyncio issues
             worker_logger.error(f"Critical error in process_symbol_batch setup/exec for {exchange_id} (Worker {os.getpid()}): {e}", exc_info=True)
             # Return error results for all symbols in this batch
             return [
                 {"symbol": s, "ohlcv_1m": None, "ohlcv_5m": None, "ohlcv_15m": None,
                  "price": None, "error": f"Batch Processing Error: {e}"}
                 for s in symbols
            ]
        finally:
            # Ensure exchange connection is closed
            if exchange:
                try:
                    await exchange.close()
                    worker_logger.debug(f"Closed exchange in worker {os.getpid()}.")
                except Exception as e:
                    worker_logger.error(f"Error closing exchange in worker {os.getpid()}: {e}")

    # Run the async batch processing within the synchronous worker function
    return asyncio.run(fetch_batch_async())

# --- Optional: Add a simple test block ---
if __name__ == '__main__':
    # This block runs only when script is executed directly (python worker_tasks.py)
    # Useful for testing the worker functions without the main app/multiprocessing
    print("Running worker_tasks.py directly for testing...")

    async def test_fetch():
        exchange_id = 'kraken' # Or binance, etc.
        test_symbol = 'BTC/USDT' # Or ETH/USDT, etc.
        timeout = 30000
        exchange = None
        try:
            config = {"enableRateLimit": True, "timeout": timeout}
            exchange = getattr(ccxt, exchange_id)(config)
            print(f"Fetching test data for {test_symbol} on {exchange_id}...")
            result = await fetch_symbol_short_term_data(exchange, test_symbol)
            print("\n--- Test Result ---")
            if result:
                print(f"Symbol: {result.get('symbol')}")
                print(f"Price: {result.get('price')}")
                print(f"Error: {result.get('error')}")
                print(f"1m candles: {len(result.get('ohlcv_1m', []))}")
                print(f"5m candles: {len(result.get('ohlcv_5m', []))}")
                print(f"15m candles: {len(result.get('ohlcv_15m', []))}")
                # print("Sample 1m:", result.get('ohlcv_1m', [])[-2:]) # Print last 2 candles
            else:
                print("No result returned.")
        except Exception as e:
            print(f"Test failed: {e}")
        finally:
            if exchange: await exchange.close(); print("Closed test exchange.")

    # Test the main worker target function (simulates what multiprocessing does)
    # Note: Replace placeholders as needed if testing the full batch function
    test_symbols = ['BTC/USDT', 'ETH/USDT']
    print(f"\nTesting process_symbol_batch with {test_symbols}...")
    # The timeframe and limit args are placeholders here as the worker ignores them
    batch_results = process_symbol_batch('kraken', test_symbols, '15m', 100, 30000)
    print("\n--- Batch Test Results ---")
    print(f"Received {len(batch_results)} results.")
    for res in batch_results:
        print(f"- {res.get('symbol')}: Price={res.get('price')}, Error='{res.get('error')}', 1m={len(res.get('ohlcv_1m',[]))}, 5m={len(res.get('ohlcv_5m',[]))}, 15m={len(res.get('ohlcv_15m',[]))}")

    # asyncio.run(test_fetch()) # Uncomment to test single symbol fetch directly

    print("\nWorker task testing complete.")