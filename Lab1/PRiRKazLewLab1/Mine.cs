using System;
/*
    • rozmiar źródła: 2000,
    • pojemność pojazdu: 200,
    • czaspozyskaniajednej jednostkiwęgla:10ms,
    • czasrozładowaniajednej jednostkiwęgla:10ms,
    • czas przejazdu między kopalnią a magazynem:10s
*/
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PRiRKazLewLab1
{
    internal class Mine
    {
        static int ore = 2000;
        static int storageOre = 0;
        static readonly int cartSize = 200;
        static readonly int oneOreMineTime = 10;
        static readonly int oneOreUnloadTime = 10;
        static readonly int travelTime = 5000;
        static SemaphoreSlim semaphoreStorage = new(1, 1);
        static SemaphoreSlim semaphoreMine = new(2, 2);
        static object lockMine = new();
        static object lockStorage = new();
        static object lockPrint = new();
        static void Main(string[] args)
        {
            PrintText($"Stan złoża: {ore} jednostek", 0);
            PrintText($"Stan magazynu: {storageOre} jednostek", 1);
            for (int j = 0; j < 5; j++)
            {
                Task[] tasks = new Task[j];
                for (int i = 0; i < tasks.Length; i++)
                {
                    var minerNum = i;
                    tasks[i] = Task.Run(() => StartMining(minerNum));
                }
                Task.WaitAll(tasks);
            }
        }
        static void StartMining(int minerNumber)
        {
            while (ore > 0)
            {
                PrintText($"Górnik {minerNumber + 1}: Idzie do kopalni...", 3+minerNumber);
                Thread.Sleep(travelTime);
                PrintText($"Górnik {minerNumber + 1}: Czeka w kopalni...", 3+minerNumber);

                int mined = 0;
                semaphoreMine.Wait();
                while (mined < cartSize && ore > 0)
                {
                    PrintText($"Górnik {minerNumber + 1}: Wydobył {mined} jednostek...", 3+minerNumber);
                    lock (lockMine)
                    {
                        Thread.Sleep(oneOreMineTime);
                        mined++;
                        ore--;
                    }
                    PrintText($"Stan złoża: {ore} jednostek", 0);
                }
                semaphoreMine.Release();
                PrintText($"Górnik {minerNumber + 1}: Idzie do magazynu...", 3+minerNumber);
                Thread.Sleep(travelTime);
                PrintText($"Górnik {minerNumber + 1}: Czeka w magazynie...", 3+minerNumber);

                semaphoreStorage.Wait();
                while (mined > 0)
                {
                    PrintText($"Górnik {minerNumber + 1}: Pozostało {mined} jednostek do wyładowania...", 3+minerNumber);
                    lock (lockStorage)
                    {
                        Thread.Sleep(oneOreUnloadTime);
                        mined--;
                        storageOre++;
                    }
                    PrintText($"Stan magazynu: {storageOre} jednostek", 1);
                }
                semaphoreStorage.Release();
            }
            PrintText($"Górnik {minerNumber + 1}: Zakończył pracę", 3+minerNumber);
        }

        private static void PrintText(string text, int lineNumber)
        {
            lock (lockPrint)
            {
                Console.SetCursorPosition(0, lineNumber);
                Console.Write(new string(' ', Console.WindowWidth));
                Console.SetCursorPosition(0, lineNumber);
                Console.Write(text);
            }
        }
    }
}
