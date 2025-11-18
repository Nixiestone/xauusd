"""
MT5 Symbol Checker & Validator
===============================
Comprehensive tool to find and test trading symbols in your MT5 terminal

Features:
- Scans all available symbols
- Identifies gold (XAU) and USD index (DXY) symbols
- Tests data availability
- Provides detailed symbol information
- Saves complete symbol list to file

Usage: python mt5_symbol_checker.py
"""

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ MetaTrader5 not installed!")
    print("   Install with: pip install MetaTrader5")
    exit()

from datetime import datetime
import pandas as pd

# ============================================================================
# SYMBOL CHECKER CLASS
# ============================================================================

class MT5SymbolChecker:
    """Check and validate MT5 symbols"""
    
    def __init__(self):
        self.symbols = []
        self.gold_symbols = []
        self.usd_symbols = []
        self.working_gold = None
        self.working_usd = None
    
    def initialize(self):
        """Initialize MT5 connection"""
        print("="*70)
        print("MT5 SYMBOL CHECKER & VALIDATOR")
        print("="*70)
        print()
        print("Connecting to MT5...")
        
        if not mt5.initialize():
            error = mt5.last_error()
            print(f"❌ Failed to connect to MT5!")
            print(f"   Error: {error}")
            print()
            print("Troubleshooting:")
            print("  1. Make sure MT5 terminal is installed and running")
            print("  2. Ensure you're logged into an account (demo is fine)")
            print("  3. Try running this script as Administrator")
            print("  4. Check if antivirus is blocking MT5")
            return False
        
        print("✓ Connected to MT5 successfully!")
        print()
        
        # Display MT5 info
        version = mt5.version()
        print(f"MT5 Terminal Information:")
        print(f"  Version: {version}")
        
        terminal_info = mt5.terminal_info()
        if terminal_info:
            print(f"  Company: {terminal_info.company}")
            print(f"  Name: {terminal_info.name}")
            print(f"  Path: {terminal_info.path}")
            print(f"  Connected: {terminal_info.connected}")
        
        print()
        
        # Get account info
        account = mt5.account_info()
        if account:
            print(f"Account Information:")
            print(f"  Login: {account.login}")
            print(f"  Server: {account.server}")
            print(f"  Broker: {account.company}")
            print(f"  Balance: ${account.balance:.2f}")
            print(f"  Currency: {account.currency}")
        
        print()
        return True
    
    def scan_all_symbols(self):
        """Scan all available symbols"""
        print("="*70)
        print("SCANNING ALL SYMBOLS")
        print("="*70)
        print()
        print("Please wait, scanning...")
        
        self.symbols = mt5.symbols_get()
        
        if not self.symbols:
            print("❌ No symbols found!")
            print("   This might mean:")
            print("   1. Your account has no symbols enabled")
            print("   2. You need to enable symbols in Market Watch")
            print("   3. Connection issue with the broker")
            return False
        
        print(f"✓ Found {len(self.symbols)} total symbols")
        print()
        
        return True
    
    def find_gold_symbols(self):
        """Find and test gold-related symbols"""
        print("="*70)
        print("SEARCHING FOR GOLD SYMBOLS")
        print("="*70)
        print()
        
        # Keywords for gold
        gold_keywords = ['XAU', 'GOLD', 'GLD', 'AU']
        
        for symbol in self.symbols:
            for keyword in gold_keywords:
                if keyword in symbol.name.upper():
                    self.gold_symbols.append(symbol)
                    break
        
        if not self.gold_symbols:
            print("❌ No gold symbols found in your MT5!")
            print()
            print("Possible reasons:")
            print("  1. Your broker doesn't offer gold trading")
            print("  2. Symbols need to be enabled in Market Watch")
            print("  3. Different naming convention (check full list)")
            print()
            return False
        
        print(f"Found {len(self.gold_symbols)} gold-related symbols:\n")
        
        # Test each gold symbol
        for i, symbol in enumerate(self.gold_symbols, 1):
            print(f"{i}. Symbol: {symbol.name}")
            print(f"   Description: {symbol.description}")
            print(f"   Visible: {'Yes' if symbol.visible else 'No'}")
            print(f"   Trade Mode: {symbol.trade_mode}")
            print(f"   Digits: {symbol.digits}")
            print(f"   Contract Size: {symbol.trade_contract_size}")
            
            # Test data availability
            rates = mt5.copy_rates_from_pos(symbol.name, mt5.TIMEFRAME_M5, 0, 10)
            
            if rates is not None and len(rates) > 0:
                print(f"   ✅ DATA AVAILABLE")
                print(f"   Latest Price: ${rates[-1][4]:.{symbol.digits}f}")
                print(f"   Latest Time: {datetime.fromtimestamp(rates[-1][0])}")
                
                # Mark as working if this is the first one
                if self.working_gold is None:
                    self.working_gold = symbol.name
                    print(f"   ⭐ RECOMMENDED - Use this symbol!")
            else:
                print(f"   ❌ NO DATA AVAILABLE")
                print(f"   Error: {mt5.last_error()}")
            
            print()
        
        return True
    
    def find_usd_symbols(self):
        """Find and test USD index symbols"""
        print("="*70)
        print("SEARCHING FOR USD INDEX SYMBOLS")
        print("="*70)
        print()
        
        # Keywords for USD index
        usd_keywords = ['DXY', 'USDX', 'DX', 'USDOLLAR', 'DOLLAR', 'USDIDX']
        
        for symbol in self.symbols:
            for keyword in usd_keywords:
                if keyword in symbol.name.upper():
                    # Exclude pairs like USDJPY, AUDUSD, etc.
                    if not any(x in symbol.name.upper() for x in ['JPY', 'EUR', 'GBP', 'AUD', 'CAD', 'CHF', 'NZD']):
                        self.usd_symbols.append(symbol)
                        break
        
        if not self.usd_symbols:
            print("❌ No USD index symbols found")
            print()
            print("This is common. Most brokers don't offer DXY.")
            print("The strategy will use EUR/USD inverse as a proxy.")
            print()
            
            # Check for EURUSD as alternative
            print("Checking for EUR/USD (will be used as DXY proxy)...")
            eurusd = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M5, 0, 10)
            
            if eurusd is not None and len(eurusd) > 0:
                print("  ✅ EUR/USD found and working")
                print(f"  Latest Price: {eurusd[-1][4]:.5f}")
                print("  ✓ Can be used as DXY proxy (inverse correlation)")
                self.working_usd = "EURUSD_PROXY"
            else:
                print("  ❌ EUR/USD also not available")
                print("  ⚠️  Strategy will run without macro sentiment filter")
            
            print()
            return False
        
        print(f"Found {len(self.usd_symbols)} USD index symbols:\n")
        
        # Test each USD symbol
        for i, symbol in enumerate(self.usd_symbols, 1):
            print(f"{i}. Symbol: {symbol.name}")
            print(f"   Description: {symbol.description}")
            print(f"   Visible: {'Yes' if symbol.visible else 'No'}")
            
            # Test data availability
            rates = mt5.copy_rates_from_pos(symbol.name, mt5.TIMEFRAME_M5, 0, 10)
            
            if rates is not None and len(rates) > 0:
                print(f"   ✅ DATA AVAILABLE")
                print(f"   Latest Price: {rates[-1][4]:.{symbol.digits}f}")
                
                if self.working_usd is None:
                    self.working_usd = symbol.name
                    print(f"   ⭐ RECOMMENDED - Use this symbol!")
            else:
                print(f"   ❌ NO DATA AVAILABLE")
            
            print()
        
        return True
    
    def generate_recommendations(self):
        """Generate final recommendations"""
        print("="*70)
        print("RECOMMENDATIONS FOR YOUR BROKER")
        print("="*70)
        print()
        
        if self.working_gold:
            print(f"✅ GOLD SYMBOL TO USE: {self.working_gold}")
            print()
            print("Update your code:")
            print(f'  symbol = "{self.working_gold}"')
            print()
            print("Or in xauusd_strategy.py, change:")
            print(f'  DEFAULT_SYMBOL = "{self.working_gold}"')
            print()
        else:
            print("❌ NO WORKING GOLD SYMBOL FOUND")
            print()
            print("Actions to take:")
            print("  1. In MT5, right-click Market Watch")
            print("  2. Select 'Symbols' or 'Show All'")
            print("  3. Find and enable gold symbols")
            print("  4. Run this script again")
            print()
            print("Alternative:")
            print("  Contact your broker to enable gold trading")
            print()
        
        if self.working_usd:
            if self.working_usd == "EURUSD_PROXY":
                print("⚠️  USD INDEX: Using EUR/USD inverse as proxy")
                print()
                print("This is normal and works fine.")
                print("The strategy automatically handles this conversion.")
            else:
                print(f"✅ USD INDEX SYMBOL TO USE: {self.working_usd}")
                print()
                print("Update your code if needed:")
                print(f'  dxy_symbol = "{self.working_usd}"')
        else:
            print("⚠️  USD INDEX: Not available")
            print()
            print("Impact: Strategy will run without macro sentiment filter")
            print("This reduces win rate slightly but strategy still works")
        
        print()
        print("="*70)
    
    def save_full_list(self):
        """Save complete symbol list to file"""
        print("="*70)
        print("SAVING COMPLETE SYMBOL LIST")
        print("="*70)
        print()
        
        filename = 'mt5_symbols_complete.txt'
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("MT5 COMPLETE SYMBOL LIST\n")
                f.write("="*70 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Symbols: {len(self.symbols)}\n\n")
                
                # Get account info
                account = mt5.account_info()
                if account:
                    f.write(f"Broker: {account.company}\n")
                    f.write(f"Server: {account.server}\n\n")
                
                f.write("="*70 + "\n")
                f.write("ALL SYMBOLS\n")
                f.write("="*70 + "\n\n")
                
                # Group by category
                forex_pairs = []
                metals = []
                indices = []
                crypto = []
                commodities = []
                stocks = []
                others = []
                
                for symbol in self.symbols:
                    name = symbol.name.upper()
                    
                    # Categorize
                    if any(x in name for x in ['EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']) and 'USD' in name:
                        forex_pairs.append(symbol)
                    elif any(x in name for x in ['XAU', 'XAG', 'GOLD', 'SILVER', 'PLAT', 'PALL']):
                        metals.append(symbol)
                    elif any(x in name for x in ['SPX', 'NDX', 'DJI', 'DAX', 'FTSE', 'NAS', 'US30', 'US500', 'DXY', 'USDX']):
                        indices.append(symbol)
                    elif any(x in name for x in ['BTC', 'ETH', 'CRYPTO', 'XRP', 'LTC']):
                        crypto.append(symbol)
                    elif any(x in name for x in ['OIL', 'WTI', 'BRENT', 'GAS', 'WHEAT', 'CORN']):
                        commodities.append(symbol)
                    elif '.' in name or any(x in name for x in ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']):
                        stocks.append(symbol)
                    else:
                        others.append(symbol)
                
                # Write categorized lists
                categories = [
                    ("PRECIOUS METALS", metals),
                    ("FOREX PAIRS", forex_pairs),
                    ("INDICES", indices),
                    ("CRYPTOCURRENCIES", crypto),
                    ("COMMODITIES", commodities),
                    ("STOCKS", stocks),
                    ("OTHER INSTRUMENTS", others)
                ]
                
                for category_name, category_symbols in categories:
                    if category_symbols:
                        f.write(f"\n{category_name} ({len(category_symbols)} symbols)\n")
                        f.write("-"*70 + "\n\n")
                        
                        for symbol in category_symbols:
                            f.write(f"Symbol: {symbol.name}\n")
                            f.write(f"  Description: {symbol.description}\n")
                            f.write(f"  Visible: {'Yes' if symbol.visible else 'No'}\n")
                            f.write(f"  Tradeable: {'Yes' if symbol.trade_mode == 4 else 'No'}\n")
                            f.write(f"  Digits: {symbol.digits}\n")
                            f.write(f"  Contract Size: {symbol.trade_contract_size}\n")
                            f.write("\n")
            
            print(f"✓ Complete symbol list saved to: {filename}")
            print(f"  Total symbols: {len(self.symbols)}")
            print(f"  Metals: {len(metals)}")
            print(f"  Forex: {len(forex_pairs)}")
            print(f"  Indices: {len(indices)}")
            print()
            
        except Exception as e:
            print(f"⚠️  Could not save file: {e}")
            print()
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        mt5.shutdown()
        print("MT5 connection closed")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    checker = MT5SymbolChecker()
    
    try:
        # Initialize MT5
        if not checker.initialize():
            return
        
        # Scan all symbols
        if not checker.scan_all_symbols():
            return
        
        # Find gold symbols
        checker.find_gold_symbols()
        
        # Find USD symbols
        checker.find_usd_symbols()
        
        # Generate recommendations
        checker.generate_recommendations()
        
        # Save complete list
        checker.save_full_list()
        
        print("="*70)
        print("SCAN COMPLETE!")
        print("="*70)
        print()
        print("Next steps:")
        print("  1. Use the recommended symbols in your strategy")
        print("  2. Review the complete list in mt5_symbols_complete.txt")
        print("  3. Enable any missing symbols in MT5 Market Watch")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Scan interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        checker.shutdown()


if __name__ == "__main__":
    main()