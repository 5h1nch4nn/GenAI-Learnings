import subprocess
import json
import shutil
import re


# Helper Fn: Check if a tool is installed
def is_tool_installed(tool_name: str) -> bool:
    """
    Check if a tool is installed on the system.

    Args:
        tool_name (str): The name of the tool.

    Returns:
        bool: True if the tool is installed, False otherwise.
    """
    return shutil.which(tool_name) is not None

# Helper Fn: Ping Host
def ping_host(host:str, count:int=2) -> str:
    """
    Pings a specified host using the system's ping utility. It verifies the availability of the ping tool before execution and handles various error cases including timeouts and network unreachability.

    Args:
        host: The hostname or IP address to ping.
        count: The number of ping packets to send. Defaults to 2.

    Returns:
        str: A JSON-formatted string containing the ping results

    Raises:
        subprocess.TimeoutExpired: If the ping command exceeds a 10-second timeout.
        Exception: For other unexpected errors during execution.
    """

    # First verify ping utility is available
    if is_tool_installed("ping"):
        try:
            # Execute ping with specified parameters
            output = subprocess.run(
                ['ping', '-c', str(count), host],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10  # 10 second timeout
            )
            
            # Process the ping results
            if output.returncode == 0:
                # Host is up and responding
                return json.dumps({
                    "host": host,
                    "status": "reachable", 
                    "details": output.stdout.strip()
                })
            else:
                # Host is down or unreachable
                return json.dumps({
                    "host": host,
                    "status": "unreachable",
                    "details": output.stdout.strip()
                })

        except subprocess.TimeoutExpired:
            return json.dumps({
                "status": "timeout",
                "details": "Ping request timed out."
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "details": str(e)
            })
    else:
        return json.dumps({
            "status": "tool not installed",
            "details": "Ping tool is not installed on the system."
        })

# Helper Fn: RustScan to get all the open ports
def rustscan_host(host: str, ports: str="1-65535") -> str:
    """
    Perform a port scan on a specified host using RustScan.

    Args:
        host: The target host IP address or domain name to scan.
        ports: Port range to scan in format "start-end". Defaults to "1-65535".

    Returns:
        str: JSON formatted string containing scan results for open ports, status and details.

    Raises:
        subprocess.TimeoutExpired: If scan operation times out
        Exception: For other unexpected errors during scan execution.
    """

    # Verify RustScan installation
    if is_tool_installed("rustscan"):
        try:
            # Execute RustScan with timeout
            result = subprocess.run(
                # ["rustscan", "-a", host, "--range", ports, "--ulimit", "5000", "-g"],
                ["rustscan", "-a", host, "--range", ports, "-g"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
            )

            # Process scan results
            if result.returncode == 0:
                # Check for permission errors
                if b"requires root privileges" in result.stderr:
                    return json.dumps({
                        "status": "error",
                        "details": "Error: RustScan requires elevated privileges. Please run as root or with sudo."
                    })

                # Process valid scan output  
                elif result.stdout != b"":
                    # Extract open ports from output
                    open_ports = re.findall(r"\[(.*?)\]", result.stdout.decode('utf-8'))[0]
                    return json.dumps({
                        "ports": open_ports,
                        "status": "success", 
                        "details": result.stdout.decode('utf-8'),
                    })
                else:
                    # No open ports found
                    return json.dumps({
                        "ports": "",
                        "status": "failure",
                        "details": f"No open ports found on {host}"
                    })

        except subprocess.TimeoutExpired:
            return json.dumps({
                "status": "timeout",
                "details": "Ping request timed out."
            })
        except Exception as e:
            return json.dumps({
                "status": "error", 
                "details": str(e),
            })

    else:
        return json.dumps({
            "status": "tool not installed",
            "details": "Rustscan tool is not installed on the system."
        })

# Helper Fn: Nmap Scan to get detailed service information
def nmapscan_host(host: str, ports: str) -> str:
    """Performs a network scan on a specified host using Nmap.
    This function executes an Nmap scan with version detection (-sV) and default script scanning (-sC)
    on the specified host and port range. It requires Nmap to be installed on the system.

    Args:
        host: The target host to scan. Can be hostname or IP address.
        ports: The ports to scan in Nmap format (e.g. "80,443" or "1-1000").

    Returns:
        str: A JSON string containing scan results with proper status and details.

    Raises:
        subprocess.TimeoutExpired: If scan exceeds 300 second timeout
        Exception: For any other unexpected errors during scan execution

    Notes:
        - Requires Nmap to be installed on the system
        - Some scan options may require root/administrator privileges
        - Default timeout is set to 300 seconds
    """
    
    # Check if Nmap is installed
    if is_tool_installed("nmap"):
        try:
            # Execute the Nmap command with a timeout
            result = subprocess.run(
                ["nmap", "-vv", "-sC", "-sV", "-p", ports, host],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300
                )
            
            # Process scan results
            if result.returncode == 0:
                if b"requires root privileges" in result.stderr:
                    # Check for permission errors
                    return json.dumps({
                        "status": "error",
                        "details": "Error: NMAP scan requires elevated privileges. Please run as root or with sudo."
                    })
                
                elif b"Host seems down" in result.stdout:
                    # Check if host is down
                    return json.dumps({
                        "status": "failure",
                        "details": result.stdout.decode('utf-8'),
                    })
                else:
                    # Return the standard output of the Nmap scan
                    return json.dumps({
                        "status": "success",
                        "details": result.stdout.decode('utf-8'),
                    })
        
        except subprocess.TimeoutExpired:
            return "Error: Nmap scan timed out. The target may be unresponsive or the scan parameters may be too broad."
        
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"


def main():
    # k = ping_host("10.10.206.119")
    k = rustscan_host("10.10.206.119")
    # k = nmapscan_host("10.10.45.123", "21,22,139,445,3333") 
    print(k)

if __name__ == "__main__":
    main()