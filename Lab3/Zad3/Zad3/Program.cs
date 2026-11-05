using System.Buffers.Binary;
using System.Net.Sockets;
using Numpy;
using Python.Runtime;
using Zad3;

const string PythonLibrary = "/usr/lib/libpython3.14.so";
const string PythonPackages =
    "/home/kazik/Projects/PRiRKazmierczakLewandowski/.venv/lib/python3.14/site-packages";

Runtime.PythonDLL = PythonLibrary;
Environment.SetEnvironmentVariable("PYTHONPATH", PythonPackages);

ClientMain();
PythonEngine.Shutdown();
return;

static void ClientMain()
{
    using var client = new TcpClient();
    client.Connect("172.20.10.2", 2040);
    var stream = client.GetStream();

    var fragment = ReceiveAll(stream);
    var processedFragment = Sobel.ApplySobel(fragment);
    SendAll(stream, processedFragment);

    client.Close();
    Console.WriteLine("Fragment przetworzony i wysłany z powrotem do serwera");
}

static NDarray ReceiveAll(NetworkStream stream)
{
    var header = new byte[4];
    stream.ReadExactly(header);
    var length = BinaryPrimitives.ReadInt32BigEndian(header);

    var data = new byte[length];
    stream.ReadExactly(data);

    NDarray buffer = np.array(data);
    using (Py.GIL())
    {
        return new NDarray(Py.Import("pickle").InvokeMethod("loads", buffer.PyObject));
    }
}

static void SendAll(NetworkStream stream, NDarray array)
{
    byte[] data;
    using (Py.GIL())
    {
        PyObject pickled = Py.Import("pickle").InvokeMethod("dumps", array.PyObject);
        var buffer = Py.Import("numpy").InvokeMethod("frombuffer", pickled, new PyString("u1"));
        data = new NDarray(buffer).GetData<byte>();
    }

    var header = new byte[4];
    BinaryPrimitives.WriteInt32BigEndian(header, data.Length);
    stream.Write(header);
    stream.Write(data);
}