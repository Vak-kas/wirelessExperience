clear variables
clc

%% 기본 세팅
SNR_dB = 20;
SNR_linear = 10.^(SNR_dB/10); %Signal Power

M = 2; % M=2라면 BPSK, M=4라면 QPSK
N =16; % 각 노드에서 생성한 랜덤한 값의 길이

nSymbols = N; %전체 비트 길이
nSymbol = nSymbols-N; %생성된 기존 비트 데이터

repeat_time =1; %FEC 에서 반복횟수
test_time = 100000; %전체 횟수

% alpha = 1; %a의 신호 세기가 e의 신호세기 보다 얼마나 더 강한지에 대한 상대적 크기
alpha_values = 0.1:0.1:2;
beta = 0; % 기준값
epsilon = 0.001; %채널 추정 오류


%% 성능 평가 변수 설정
true_probability = zeros(1, length(SNR_dB));
false_probability = zeros(1, length(SNR_dB));

% A->B AR 요청
for i = 1:1:length(alpha_values)

    alpha = alpha_values(i);


    % 반복 실험을 통해 평균 탐지 확률 및 오탐지 확률 계산
    true_detection_sum = 0; % 탐지 성공 횟수 누적
    false_detection_sum = 0; % 오탐지 횟수 누적

    for j = 1:test_time
        %% 실험 준비
        %h1은 A와 B사이 (정상 루트)
        h1 = sqrt(1/2)*(randn(1, (nSymbols*repeat_time/log2(M))) + 1j*randn(1, (nSymbols*repeat_time/log2(M))) ); 
        
        %h2는 B와 Eve 사이(도청 루트)
        h2 = sqrt(1/2)*(randn(1, (nSymbols*repeat_time/log2(M))) + 1j*randn(1, (nSymbols*repeat_time/log2(M))) ); 
        
        
        %각 노드들에서 받아들이는 노이즈 값 (a, b, e)
        noise_a = sqrt(1/2)*(randn(1, (nSymbols*repeat_time/log2(M))) + 1j*randn(1, (nSymbols*repeat_time/log2(M))) ); 
        noise_b = sqrt(1/2)*(randn(1, (nSymbols*repeat_time/log2(M))) + 1j*randn(1, (nSymbols*repeat_time/log2(M))) ); 
        noise_e = sqrt(1/2)*(randn(1, (nSymbols*repeat_time/log2(M))) + 1j*randn(1, (nSymbols*repeat_time/log2(M))) ); 
        
        %% 실험 과정 1
        % A와 B가 각각 무선 채널 예측하여 키 값 생성

        h_a = h1 +epsilon*noise_a; % A측에서 예측하는 h1
        h_b = h1 + epsilon*noise_b; % B측에서 예측하는 h1
        h_b2 = h2 + epsilon*noise_b; % B측에서 예측하는 h2
        h_e = h2 + epsilon*noise_e; % E측에서 예측하는 h2

        % A측에서 무선채널을 보고 생성한 key값
        key1 = zeros(1, length(h1)); 
        key1(abs(h_a).^2>beta) = 1 ; %B측에서 예측하는 h2

        % B측에서 무선채널을 보고 생성한 key값
        key2= zeros(1, length(h1));
        key2(abs(h_b).^2>beta) = 1;
        

        % B측에서 도청자와의 무선채널을 보고 생성한 key값
        key4 = zeros(1, length(h2));
        key4(abs(h_b2).^2>beta) =1;

        % e측에서 도청자와의 무선채널을 보고 생성한 key값
        key3 = zeros(1, length(h2));
        key3(abs(h_e).^2>beta) =1;
        
            
        %% 실힘 과정 2
        % B->A B에서 생성한 임의의 데이터를 A측으로 보내기
        
        bit_stream = randi([0 1], 1, nSymbol); %비트 데이터 생성
        random_value = randi([0 1], 1, N); %생성된 랜덤 value
        
        bit_data = horzcat(bit_stream, random_value); %전체 비트 + 랜덤 값
        
        fec_bit_data = REP_FEC(bit_data, repeat_time);
        
        %변조
        if M==2
            modulated_symbol = BPSK_Mapper(fec_bit_data);
        elseif M==4
            modulated_symbol = QPSK_Mapper(fec_bit_data);
        end
        
        
        %전송
        transmit_power = SNR_linear; % 출력세기 (y(n))
        transmission_symbol = sqrt(transmit_power)*modulated_symbol.*h1 + noise_a; %수신측에서 전송받은 심볼
        
        
        %% 실험 과정 3
        % A가 B로부터 신호 받은 것을 수신
        
        received_symbol = transmission_symbol./h1; %equalizer
        
        
        if M==2
            fec_recovered_bit_data = BPSK_Demapper(received_symbol);
        elseif M==4
            fec_recovered_bit_data = QPSK_Demapper(received_symbol);
        end
        
        recovered_bit_data = FEC_check(fec_recovered_bit_data, repeat_time);
       
        
        %% 실험과정 4
        % A가 생성한 키 값을 B측으로 전송, 받은 값에 XOR 연산 후 전송
        
        bit_stream = randi([0 1], 1, nSymbol); %비트 데이터 생성
        % key_a = randi([0 1], 1, N); %맨 뒤에 붙는 키 값
        key_a = key1;

        
        tmp = recovered_bit_data(nSymbol+1:end);
        xor_key_a = xor(key_a, tmp);
        
        
        key_bit_data = horzcat(bit_stream, xor_key_a);%전체 비트 + 키 값
        
        fec_key_bit_data = REP_FEC(key_bit_data, repeat_time);
        
        
        
        if M==2
            modulated_symbol = BPSK_Mapper(fec_key_bit_data);
        elseif M==4
            modulated_symbol = QPSK_Mapper(fec_key_bit_data);
        end
        
        transmit_power = SNR_linear; % 출력세기 (y(n))
        transmission_symbol = sqrt(transmit_power*alpha) * modulated_symbol.*h1 + noise_b; 
        
        %% 실험과정 5
        %B가 A로 부터 받은 신호 수신 후 다시 기존에 있던 bit_data(b가 생성한 랜덤한 값)로 xor연산
        received_symbol = transmission_symbol./h1; %equalizer
        
        
        if M==2
            fec_recovered_bit_data = BPSK_Demapper(received_symbol);
        elseif M==4
            fec_recovered_bit_data = QPSK_Demapper(received_symbol);
        end
        
        recovered_bit_data = FEC_check(fec_recovered_bit_data, repeat_time);
        tmp = recovered_bit_data(nSymbol+1:end);
        recovered_key = xor(random_value, tmp);
        %% 정상 탐지 확률 계산
        true_detection_sum = true_detection_sum + sum(recovered_key == key2) / N; % 키 일치 비율
        
        
        %% 도청자 입장


        bit_stream = randi([0 1], 1, nSymbol); %비트 데이터 생성
        % key_a = randi([0 1], 1, N); %맨 뒤에 붙는 키 값
        key_c = key3;

        
        tmp = recovered_bit_data(nSymbol+1:end);
        xor_key_c = xor(key_c, tmp);
        
        
        key_bit_data = horzcat(bit_stream, xor_key_c);%전체 비트 + 키 값
        
        fec_key_bit_data = REP_FEC(key_bit_data, repeat_time);
        
        
        
        if M==2
            modulated_symbol = BPSK_Mapper(fec_key_bit_data);
        elseif M==4
            modulated_symbol = QPSK_Mapper(fec_key_bit_data);
        end
        
        transmit_power = SNR_linear; % 출력세기 (y(n))
        transmission_symbol = sqrt(transmit_power) * modulated_symbol.*h2 + noise_b; 
        
        received_symbol = transmission_symbol./h1; %equalizer
        
        
        if M==2
            fec_recovered_bit_data = BPSK_Demapper(received_symbol);
        elseif M==4
            fec_recovered_bit_data = QPSK_Demapper(received_symbol);
        end
        
        recovered_bit_data = FEC_check(fec_recovered_bit_data, repeat_time);
        
        tmp = recovered_bit_data(nSymbol+1:end);
        recovered_key = xor(random_value, tmp);
    
        %% 오탐지 확률 계산
        false_detection_sum = false_detection_sum + sum(recovered_key == key2) / N;
    end

    %% 평균 탐지 확률 및 오탐지 확률 계산
    true_probability(i) = true_detection_sum / test_time; % 탐지 확률
    false_probability(i) = false_detection_sum / test_time; % 오탐지 확률
end

%% 결과 그래프 그리기
figure(1);
hold on;
%plot(alpha_values, false_probability, '-rx', 'DisplayName', 'false alarm (P_f)');


%plot(alpha_values, true_probability, '-go', 'DisplayName', 'snr = -10 탐지 확률 (P_d)');
%plot(alpha_values, true_probability, '-yo', 'DisplayName', 'snr = 5 탐지 확률 (P_d)');

%plot(alpha_values, true_probability, '-ko', 'DisplayName', 'snr = 10 탐지 확률 (P_d)');
plot(alpha_values, true_probability, '-bo', 'DisplayName', 'snr = 20 탐지 확률 (P_d)');



xlabel('Alpha 값 (상대적 신호 강도 비율)');
ylabel('Probability');
legend show;
title('Alpha 값에 따른 탐지 및 오탐지 확률');
grid on;





%% functions

% BPSK_Mapping
function [modulated_symbol] = BPSK_Mapper(data)
    modulated_symbol = zeros(1, length(data));

    modulated_symbol(data==1) = (1+1j)/sqrt(2);
    modulated_symbol(data==0) = (-1-1j)/sqrt(2);
end

% BPSK_DeMapping
function [recovered_data] = BPSK_Demapper(received_symbol)
    recovered_data = zeros(1, length(received_symbol));

    recovered_data(real(received_symbol) + imag(received_symbol) > 0) = 1;
    recovered_data(real(received_symbol) + imag(received_symbol) < 0) = 0;
end


%QPSK 맵핑 함수
function [QPSK_Symbol] =  QPSK_Mapper(Data)
    
    N = length(Data); % 길이

    QPSK_Symbol = zeros(1, N/2);
    for i= 1:N/2
        two_bit = [Data(2*i-1) Data(2*i)];
        % disp(two_bit)
    
        if two_bit == [0 0]
            QPSK_Symbol(i) = sqrt(1/2) + sqrt(1/2)*j;
    
        elseif two_bit == [0 1]
            QPSK_Symbol(i) = -sqrt(1/2) + sqrt(1/2)*j;
    
        elseif two_bit == [1 1]
            QPSK_Symbol(i) = -sqrt(1/2) -sqrt(1/2)*j;
    
        else
            QPSK_Symbol(i) = sqrt(1/2) -sqrt(1/2)*j;
    
        end
    end

end

%QPSK 디맵핑 함수
function [x_hat] = QPSK_Demapper(r)
    L = length(r);
    x_hat = zeros(1, 2*L);

    for n = 1:L
        if real(r(n)) >=0 && imag(r(n)) >= 0
            x_hat(2*(n-1)+1:2*n) = [0 0];
        elseif real(r(n)) <0 && imag(r(n)) >= 0
            x_hat(2*(n-1)+1:2*n) = [0 1];
        elseif real(r(n)) <0 && imag(r(n)) < 0
            x_hat(2*(n-1)+1:2*n) = [1 1];
        else
             x_hat(2*(n-1)+1:2*n) = [1 0];
        end

    end
    % disp(x_hat)
end
%% fec

function [Repeat_bit_data]  = REP_FEC(bit_data, Repeat_time)
    Repeat_bit_data = repmat(bit_data, 1, Repeat_time);
end

function [bit_data] = FEC_check(Repeat_bit_data, Repeat_time)
    bit_data = zeros(1, length(Repeat_bit_data)/Repeat_time);
    nSymbol = length(bit_data);
    for i = 1:nSymbol
        s = 0;
        for j  = 1:Repeat_time
            s = s + Repeat_bit_data(nSymbol*(j-1)+i);    
        end
        bit_data(i) = round((s/Repeat_time));
    end

end



